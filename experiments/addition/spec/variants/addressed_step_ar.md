# Addition — Addressed Per-Step Autoregressive Variant (`addressed_step_ar`)

The `addressed_step_ar` variant (`experiments/addition/variants/addressed_step_ar/`) combines the step-by-step harness of [`step_ar`](step_ar.md) with the explicit base-`B` column addresses of [`addressed_ar`](addressed_ar.md).

Why `addressed_step_ar`:
- **Against [`step_ar`](step_ar.md)**: holds the step-by-step harness and `SubspaceLMHead` fixed while adding explicit base-`B` column addresses `addr(j)` to every digit and to the step-`k` query, replacing positional counting with content-based address matching (`addr(j) == addr(k)`).
- **Against [`addressed_ar`](addressed_ar.md)**: holds the addressed representation and `SubspaceLMHead` fixed while offloading inter-column state updates (`c_k <- c_{k+1}`) and column-address counting (`addr(k)` supplied in the step-`k` query by the harness) out of the model's trace.

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - `SUM_FIRST` (`"sum_first"`): emits `o_k`, then `c_{k+1}`, then `flag_k`.
   - `CARRY_FIRST` (`"carry_first"`): emits `c_{k+1}`, then `o_k`, then `flag_k`.
2. **`use_query_tokens: bool`** (`False` default):
   - `False`: after `[IDX] addr(k)` at the end of the step-`k` prompt, the model emits the 3 targets `(o_k, c_{k+1}, flag_k)` (or `(c_{k+1}, o_k, flag_k)`) directly in 3 consecutive positions.
   - `True`: the harness inserts query tokens (`[S]`, `[C]`, `[F]`) before each of the 3 predictions (`[IDX] addr(k) [S] o_k [C] c_{k+1} [F] flag_k` under `sum_first`, or `[IDX] addr(k) [C] c_{k+1} [S] o_k [F] flag_k` under `carry_first`), supervising only the 3 target positions after `[S]`, `[C]`, and `[F]`.
3. **`include_alphabet: bool`** (`True` default):
   - `True`: includes the explicit `s_0 ... s_{B-1} [SEP]` alphabet section after `[BOS]`.
   - `False` *(zero-preamble mode)*: omits `s_0 ... s_{B-1} [SEP]` from the prompt and requires operand lengths `L_a, L_b >= B` so every symbol `s_0, ..., s_{B-1}` appears in ascending order as the single-digit column addresses `addr(0) .. addr(B - 1)` of the operands. The model must infer both `B` and the alphabet order from the operand addresses alone.

## 1. Sequence Layout

Using the base-`B` column address encoding `addr(j, alphabet)` ([`addressed_ar.md`](addressed_ar.md#base-b-column-addresses-addrj-alphabet)), at column `k` (`0 <= k <= K`) the prompt contains:
- `[BOS]` followed by `s_0 ... s_{B-1} [SEP]` when `include_alphabet=True` (omitted when `include_alphabet=False`),
- Operand `a`: `[IDX] addr(0) a_0 ... [IDX] addr(L_a-1) a_{L_a-1} [SEP]`,
- Operand `b`: `[IDX] addr(0) b_0 ... [IDX] addr(L_b-1) b_{L_b-1} [SEP]`,
- Incoming carry `c_k` at address `addr(k)`: `[IDX] addr(k) c_k [SEP]`,
- Output history `o_0, ..., o_{k-1}`: `[IDX] addr(0) o_0 ... [IDX] addr(k-1) o_{k-1} [SEP]`,
- Step-`k` index hint: `[IDX] addr(k)` (followed by `o_k c_{k+1}` when `use_query_tokens=False`, or `[S] o_k [C] c_{k+1} [F]` when `use_query_tokens=True`).

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet *(if `include_alphabet=True`)* | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter *(if `include_alphabet=True`)* | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` entry `j` (`0 <= j < L_a`) | `[IDX]`, `addr(j)`, `a_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_A` (`2`) | `j` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` entry `j` (`0 <= j < L_b`) | `[IDX]`, `addr(j)`, `b_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_B` (`3`) | `j` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Incoming carry | `[IDX]`, `addr(k)`, `c_k` | `SPECIAL` (`0`), `ADDRESS` (`6`), `CARRY` (`4`) | `k` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output history entry `j` (`0 <= j < k`) | `[IDX]`, `addr(j)`, `o_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OUTPUT` (`5`) | `j` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Step `k` index hint | `[IDX]`, `addr(k)` | `SPECIAL` (`0`), `ADDRESS` (`6`) | `k` |
| Step `k` query tokens *(if `use_query_tokens=True`)* | `[S]`, `[C]`, `[F]` | `TokenType.SPECIAL` (`0`) | `k` |
| Step `k` sum digit | `o_k` | `TokenType.OUTPUT` (`5`) | `k` |
| Step `k` outgoing carry | `c_{k+1}` | `TokenType.CARRY` (`4`) | `k + 1` |

*(Under `carry_first`, the outgoing carry `c_{k+1}` and sum digit `o_k`—along with their preceding `[C]` and `[S]` query tokens when `use_query_tokens=True`—appear in the order carry then sum, keeping the exact same `token_type_ids` and `position_ids`.)*

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `output = ['2', '1', '0', '1']` (`1012`), `include_alphabet=True`, `use_query_tokens=False`):**

```
k = 0:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [IDX] 0 7 [IDX] 1 8 [IDX] 2 9 [SEP] [IDX] 0 5 [IDX] 1 2 [SEP] [IDX] 0 0 [SEP] [SEP] [IDX] 0
        out (sum_first):   2 1 [FLAG_CONTINUE]
        out (carry_first): 1 2 [FLAG_CONTINUE]

k = 1:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [IDX] 0 7 [IDX] 1 8 [IDX] 2 9 [SEP] [IDX] 0 5 [IDX] 1 2 [SEP] [IDX] 1 1 [SEP] [IDX] 0 2 [SEP] [IDX] 1
        out (sum_first):   1 1 [FLAG_CONTINUE]
        out (carry_first): 1 1 [FLAG_CONTINUE]

k = 2:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [IDX] 0 7 [IDX] 1 8 [IDX] 2 9 [SEP] [IDX] 0 5 [IDX] 1 2 [SEP] [IDX] 2 1 [SEP] [IDX] 0 2 [IDX] 1 1 [SEP] [IDX] 2
        out (sum_first):   0 1 [FLAG_CONTINUE]
        out (carry_first): 1 0 [FLAG_CONTINUE]

k = 3:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [IDX] 0 7 [IDX] 1 8 [IDX] 2 9 [SEP] [IDX] 0 5 [IDX] 1 2 [SEP] [IDX] 3 1 [SEP] [IDX] 0 2 [IDX] 1 1 [IDX] 2 0 [SEP] [IDX] 3
        out (sum_first):   1 0 [FLAG_FINISHED]
        out (carry_first): 0 1 [FLAG_FINISHED]
```

**Example 2 (`alphabet = ['q', '#', '7']` (`B = 3`: `q = 0`, `# = 1`, `7 = 2`), `a = ['7', '#', 'q']` (`L_a = 3 >= B`), `b = ['#', 'q', '#']` (`L_b = 3 >= B`), `include_alphabet=False`, `use_query_tokens=True`):**

```
k = 0:  in (sum_first):    [BOS] [IDX] q 7 [IDX] # # [IDX] 7 q [SEP] [IDX] q # [IDX] # q [IDX] 7 # [SEP] [IDX] q q [SEP] [SEP] [IDX] q [S] -> q, [C] -> #, [F] -> [FLAG_CONTINUE]
```

## 3. Supervision & Rollout

Let `(t_0, t_1, flag_k)` be `(o_k, c_{k+1}, flag_k)` when `step_order == SUM_FIRST`, or `(c_{k+1}, o_k, flag_k)` when `step_order == CARRY_FIRST`, and let `(Q_0, Q_1, Q_2)` be `([S], [C], [F])` when `step_order == SUM_FIRST` or `([C], [S], [F])` when `step_order == CARRY_FIRST`:

- **Teacher-forced step examples**: at step `k` (`0 <= k <= K`), let `P_k` be the length of the step-`k` prompt ending at `[IDX] addr(k)`:
  - When `use_query_tokens=False`: `example` appends `(t_0, t_1)` to the step-`k` prompt (length `P_k + 2`), and `labels` is `(IGNORE_INDEX,) * (P_k - 1) + (t_0, t_1, flag_k)`.
  - When `use_query_tokens=True`: `example` appends `(Q_0, t_0, Q_1, t_1, Q_2)` to the step-`k` prompt (length `P_k + 5`), and `labels` has `t_0` at `P_k` (the position of `Q_0`), `t_1` at `P_k + 2` (the position of `Q_1`), `flag_k` at `P_k + 4` (the position of `Q_2`), and `IGNORE_INDEX = -100` everywhere else (harness-inserted query tokens `Q_0, Q_1, Q_2` are never prediction targets).
  During training, each batch item is drawn by generating a fresh problem and sampling a single column index `k` with terminal-step oversampling `p_terminal` ([`task.md` §1](../task.md#alphabets-problem-sampling--generalisation-yessiralphabet)); during evaluation, all `K + 1` steps of each problem are evaluated.
- **Rollout (`rollout_addressed_step_ar`)**: Free-running greedy execution starts at `k = 0` with `carry = s_0` and `output = ()`. At each step `k` (up to `max(L_a, L_b) + 1` steps):
  1. Starting from the step-`k` prompt ending at `[IDX] addr(k)`, the harness runs 3 greedy `argmax` forward passes through the model, reading each next token from `SubspaceLMHead` (inserting `Q_0`, `Q_1`, `Q_2` before each pass when `use_query_tokens=True`) to decode `t_0`, `t_1`, and `flag_k`, and unpacks `(o_k, c_{k+1})` according to `step_order`.
  2. If `flag_k == FLAG_FINISHED`, rollout stops and returns the emitted `(o_k, c_{k+1}, flag_k)` step tuples; otherwise the harness discards the transient step tokens, updates `carry = c_{k+1}`, appends `o_k` to `output`, and advances to `k + 1`.

## 4. Evaluation Metrics (`evaluate_addressed_step_ar`)

Given teacher-forced greedy predictions at the 3 supervised positions of each step `k` in `0 .. K` (unpacked according to `step_order` into `(pred_o[k], pred_c[k], pred_flag[k])`), `evaluate_addressed_step_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where every step `k` in `0 .. K` predicts `pred_o[k] == o_k`, `pred_c[k] == c_{k+1}`, and `pred_flag[k] == flag_k`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature halt (`pred_flag[k] == FLAG_CONTINUE` for every `k < K`) and halts on the final column (`pred_flag[K] == FLAG_FINISHED`).
