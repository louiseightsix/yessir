# Addition — Per-Step Autoregressive Variant (`step_ar`)

The `step_ar` variant (`experiments/addition/variants/step_ar/`) frames [the `addition` task](../task.md) as sequential per-step next-token prediction through `SubspaceLMHead`, driven by the same step-by-step harness as [`cls`](cls.md).

Holding the per-step harness fixed (updating `c_k` and `o_0, ..., o_{k-1}` between columns and discarding transient step tokens), `step_ar` replaces `cls`'s 3-head `ParallelClassificationHead` with 3 sequential next-token predictions through the parameter-free `SubspaceLMHead`.

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - **`StepOrder.SUM_FIRST` (`"sum_first"`, default)**: emits `(o_k, c_{k+1}, flag_k)`. The model must compute the modular residue `o_k = alphabet[(v_a + v_b + v_c) % B]` directly on the first prediction, before emitting `c_{k+1}`.
   - **`StepOrder.CARRY_FIRST` (`"carry_first"`)**: emits `(c_{k+1}, o_k, flag_k)`. The model first predicts the monotone carry threshold (`c_{k+1} = s_1` if `v_a + v_b + v_c >= B` else `s_0`); once `c_{k+1}` is appended to the sequence, the second prediction `o_k` can condition on `c_{k+1}` (a conditional subtraction by `B`).
2. **`use_query_tokens: bool`** (`False` default):
   - **`False`**: after the step-trigger `[SEP]`, the model emits the 3 targets `(o_k, c_{k+1}, flag_k)` (or `(c_{k+1}, o_k, flag_k)`) directly in 3 consecutive positions.
   - **`True`**: replaces the step-trigger `[SEP]` with explicit harness-inserted query triggers (`[S]`, `[C]`, `[F]`; see [`step_fork.md`](step_fork.md#1-sequence-layout--query-triggers)) before each of the 3 predictions (`[S] o_k [C] c_{k+1} [F] flag_k` under `sum_first`, or `[C] c_{k+1} [S] o_k [F] flag_k` under `carry_first`), supervising only the 3 target positions after `[S]`, `[C]`, and `[F]`.

Between columns, the harness behaves identically under all settings: it discards the transient step tokens, updates `carry = c_{k+1}`, and appends `o_k` to `output`.

## 1. Sequence Layout

At column `k` (`0 <= k <= K`), the prompt contains the shared problem prefix ([`task.md` §2](../task.md#2-shared-problem-prefix--embedding-modes-experimentsadditioninputspy)), the incoming carry `c_k`, the sum digits `o_0, ..., o_{k-1}` emitted by earlier columns, and a trailing `[SEP]` delimiter after which the model emits the current column's 3 tokens:

```
sum_first:    [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [SEP]  o_k  c_{k+1}
carry_first:  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [SEP]  c_{k+1}  o_k
```

Why a `[SEP]` delimiter separates `o_0, ..., o_{k-1}` from the current step's tokens:
- Without a delimiter after `o_0, ..., o_{k-1}`, under `sum_first` the prefix ending in `o_0` at step `k = 0` (`... [SEP] c_0 [SEP] o_0`, which must predict `c_1`) would be token-for-token identical to the prompt at step `k = 1` (`... [SEP] c_1 [SEP] o_0`, which must predict `o_1`) whenever `c_0 == c_1` (e.g. `c_0 = c_1 = s_0`). Placing `[SEP]` after `o_0, ..., o_{k-1}` marks the boundary between completed history (`o_0, ..., o_{k-1}`) and the active step's predictions in all three embedding modes and under both step orders.

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` | `a_0, ..., a_{L_a-1}` | `TokenType.OPERAND_A` (`2`) | `0, 1, ..., L_a - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` | `b_0, ..., b_{L_b-1}` | `TokenType.OPERAND_B` (`3`) | `0, 1, ..., L_b - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Incoming carry | `c_k` | `TokenType.CARRY` (`4`) | `k` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output history | `o_0, ..., o_{k-1}` | `TokenType.OUTPUT` (`5`) | `0, 1, ..., k - 1` |
| Step trigger | `[SEP]` | `TokenType.SPECIAL` (`0`) | `k` |
| Step sum digit | `o_k` | `TokenType.OUTPUT` (`5`) | `k` |
| Step outgoing carry | `c_{k+1}` | `TokenType.CARRY` (`4`) | `k + 1` |

*(Under `carry_first`, the final two rows appear in the order `c_{k+1}` then `o_k`, keeping the same `token_type_ids` and `position_ids`.)*

Position coupling at column `k`:
- Each historical sum digit `o_j` (`0 <= j < k`) sits at `position_id = j`.
- The step-trigger `[SEP]` and `o_k` both sit at `position_id = k`, matching `a_k` (`TokenType.OPERAND_A`), `b_k` (`TokenType.OPERAND_B`), and `c_k` (`TokenType.CARRY`) at `position_id = k` with zero offset.
- The emitted outgoing carry `c_{k+1}` sits at `position_id = k + 1` (the index of the next column it feeds into, keeping `(TokenType.CARRY, k + 1)` distinct from the incoming carry `c_k` at `(TokenType.CARRY, k)`).

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `out = ['2', '1', '0', '1']` (`1012`)):**

```
k = 0:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 0 [SEP] [SEP]
        out (sum_first):   2 1 [FLAG_CONTINUE]
        out (carry_first): 1 2 [FLAG_CONTINUE]

k = 1:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 [SEP]
        out (sum_first):   1 1 [FLAG_CONTINUE]
        out (carry_first): 1 1 [FLAG_CONTINUE]

k = 2:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 [SEP]
        out (sum_first):   0 1 [FLAG_CONTINUE]
        out (carry_first): 1 0 [FLAG_CONTINUE]

k = 3:  in:                [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 0 [SEP]
        out (sum_first):   1 0 [FLAG_FINISHED]
        out (carry_first): 0 1 [FLAG_FINISHED]
```

**Example 2 (`alphabet = ['q', '#', '7']`, `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `out = ['q', '7']` (`6`)):**

```
k = 0:  in:                [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] q [SEP] [SEP]
        out (sum_first):   q # [FLAG_CONTINUE]
        out (carry_first): # q [FLAG_CONTINUE]

k = 1:  in:                [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] # [SEP] q [SEP]
        out (sum_first):   7 q [FLAG_FINISHED]
        out (carry_first): q 7 [FLAG_FINISHED]
```

## 3. Supervision & Rollout

Let `(t_0, t_1, flag_k)` be `(o_k, c_{k+1}, flag_k)` when `step_order == SUM_FIRST`, or `(c_{k+1}, o_k, flag_k)` when `step_order == CARRY_FIRST`:

- **Teacher-forced step examples**: at step `k` (`0 <= k <= K`), let `P_k` be the length of the step-`k` prompt ending at the step-trigger `[SEP]`:
  - `example` appends `(t_0, t_1)` to the step-`k` prompt (length `P_k + 2`).
  - `labels` is `(IGNORE_INDEX,) * (P_k - 1) + (t_0, t_1, flag_k)` (length `P_k + 2`), so `SubspaceLMHead` + `next_token_cross_entropy` supervise the 3 step predictions (`[SEP] -> t_0`, `t_0 -> t_1`, `t_1 -> flag_k`) in one forward pass per step while masking all earlier prompt positions with `IGNORE_INDEX = -100`.
  During training, each batch item is drawn by generating a fresh problem and sampling a single column index `k` with terminal-step oversampling `p_terminal` ([`task.md` §1](../task.md#alphabets-problem-sampling--generalisation-yessiralphabet)); during evaluation, all `K + 1` steps of each problem are evaluated.
- **Rollout (`rollout_step_ar`)**: Free-running greedy execution starts at `k = 0` with `carry = s_0` and `output = ()`. At each step `k` (up to `max(L_a, L_b) + 1` steps):
  1. Starting from the step-`k` prompt ending at `[SEP]`, the harness runs 3 greedy `argmax` forward passes through `SubspaceLMHead` to emit `t_0`, `t_1`, and `flag_k` one token at a time, and unpacks `(o_k, c_{k+1})` according to `step_order`.
  2. If `flag_k == FLAG_FINISHED`, rollout stops and returns the emitted `(o_k, c_{k+1}, flag_k)` step tuples; otherwise the harness discards the transient step tokens, updates `carry = c_{k+1}`, appends `o_k` to `output`, and advances to `k + 1`.

## 4. Evaluation Metrics (`evaluate_step_ar`)

Given teacher-forced greedy predictions at the 3 supervised positions of each step `k` in `0 .. K` (unpacked according to `step_order` into `(pred_o[k], pred_c[k], pred_flag[k])`), `evaluate_step_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where every step `k` in `0 .. K` predicts `pred_o[k] == o_k`, `pred_c[k] == c_{k+1}`, and `pred_flag[k] == flag_k`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature halt (`pred_flag[k] == FLAG_CONTINUE` for every `k < K`) and halts on the final column (`pred_flag[K] == FLAG_FINISHED`).
