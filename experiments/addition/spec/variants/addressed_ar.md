# Addition — Addressed Full-Trace Autoregressive Variant (`addressed_ar`)

The `addressed_ar` variant (`experiments/addition/variants/addressed_ar/`) frames [the `addition` task](../task.md) as full-trace autoregressive next-token prediction through `SubspaceLMHead`, where every operand digit and every trace step is paired with its **explicit column index written as a base-`B` numeral in the prompt's own alphabet**.

Why explicit in-sequence column addresses:
- In [`ar`](ar.md), aligning column `k` relies either on sinusoidal `position_ids` (`TOKEN_TYPE_PE`, which encounters unseen sinusoidal phases when operand lengths exceed the training range `L > L_train`) or on counting `k` tokens from `[SEP]` under causal masking (`TOKEN_TYPE_NOPE` / `NOPE`, where attention softmax weights dilute over long sequences).
- Writing each digit's column index `j` directly in the token stream in base `B` allows the model to increment the column counter `addr(k)` in its trace and retrieve `a_k` and `b_k` by content-based address matching (`addr(j) == addr(k)`), without sinusoidal coordinate drift or long-range token counting.

### Base-`B` column addresses (`addr(j, alphabet)`)

Pairing each digit with its column index in the sequence is what Zhou et al. (2023, "What Algorithms Can Transformers Learn? A Study in Length Generalization") call **index hints**; here each hint is written as a base-`B` numeral in the prompt's own alphabet.

For any non-negative column index `j >= 0` and ordered alphabet `alphabet = [s_0, ..., s_{B-1}]` of base `B = len(alphabet)`:
- `addr(0, alphabet) = (s_0,)`.
- For `j > 0`, `addr(j, alphabet)` is the non-empty tuple of base-`B` digits of `j` in least-significant-first (`LSD -> MSD`) order:
  - For `0 <= j < B`, `addr(j, alphabet) = (s_j,)` (a single symbol).
  - For `B <= j < B^2`, `addr(j, alphabet) = (s_{j % B}, s_{j // B})` (e.g. column `24` in base 10 has `addr(24) = ('4', '2')`; column `5` in base 2 has `addr(5) = (s_1, s_0, s_1)`).

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - `SUM_FIRST` (`"sum_first"`): each column `k` emits `o_k` before `c_{k+1}`.
   - `CARRY_FIRST` (`"carry_first"`): each column `k` emits `c_{k+1}` before `o_k`.
2. **`use_query_tokens: bool`** (`False` default):
   - `False`: each column `k` in the trace emits `[IDX] addr(k) o_k c_{k+1}` (or `[IDX] addr(k) c_{k+1} o_k` under `carry_first`).
   - `True`: each value token in the trace is preceded by its query token (`[S]` for `o_k`, `[C]` for `c_{k+1}`): `[IDX] addr(k) [S] o_k [C] c_{k+1}` under `sum_first`, or `[IDX] addr(k) [C] c_{k+1} [S] o_k` under `carry_first`.
3. **`include_alphabet: bool`** (`True` default):
   - `True`: includes the explicit `s_0 ... s_{B-1} [SEP]` alphabet section after `[BOS]`.
   - `False` *(zero-preamble mode)*: omits `s_0 ... s_{B-1} [SEP]` from the prompt and requires operand lengths `L_a, L_b >= B` so every symbol `s_0, ..., s_{B-1}` appears in ascending order as the single-digit column addresses `addr(0) .. addr(B - 1)` of the operands.

After column `K`, the trace terminates with `[EOS]`.

## 1. Sequence Layout

Major sections (`alphabet`, `a`, `b`, and the output `trace`) are separated by `[SEP]`, matching [`ar`](ar.md). Within `a`, `b`, and `trace`, each column entry begins with `[IDX]` (`AdditionToken.IDX = 0xFB`, in `vocab.EXPERIMENT_SPECIAL_RANGE` alongside the query tokens) followed by its base-`B` address `addr(j)` and value token(s), so multi-digit addresses (`j >= B`) are unambiguously delimited in all embedding modes:

```
Prefix:
  [BOS]  s_0 ... s_{B-1}
  [SEP]  [IDX] addr(0) a_0  [IDX] addr(1) a_1 ... [IDX] addr(L_a-1) a_{L_a-1}
  [SEP]  [IDX] addr(0) b_0  [IDX] addr(1) b_1 ... [IDX] addr(L_b-1) b_{L_b-1}
  [SEP]

Trace (sum_first):
  [IDX] addr(0) o_0 c_1  [IDX] addr(1) o_1 c_2 ... [IDX] addr(K) o_K c_{K+1}

Trace (carry_first):
  [IDX] addr(0) c_1 o_0  [IDX] addr(1) c_2 o_1 ... [IDX] addr(K) c_{K+1} o_K
```

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet *(if `include_alphabet=True`)* | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter *(if `include_alphabet=True`)* | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` entry `j` (`0 <= j < L_a`) | `[IDX]`, `addr(j)`, `a_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_A` (`2`) | `j` (for all tokens in entry `j`) |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` entry `j` (`0 <= j < L_b`) | `[IDX]`, `addr(j)`, `b_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_B` (`3`) | `j` (for all tokens in entry `j`) |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Trace column `k` marker (`0 <= k <= K`) | `[IDX]` | `TokenType.SPECIAL` (`0`) | `k` |
| Trace column `k` address | `addr(k)` | `TokenType.ADDRESS` (`6`) | `k` |
| Trace column `k` query tokens *(if `use_query_tokens=True`)* | `[S]`, `[C]` | `TokenType.SPECIAL` (`0`) | `k` |
| Trace column `k` sum digit | `o_k` | `TokenType.OUTPUT` (`5`) | `k` |
| Trace column `k` outgoing carry | `c_{k+1}` | `TokenType.CARRY` (`4`) | `k + 1` |

*(Under `carry_first`, column `k` emits `c_{k+1}` before `o_k`—and `[C] c_{k+1}` before `[S] o_k` when `use_query_tokens=True`—keeping the exact same `token_type_ids` and `position_ids`.)*

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `output = ['2', '1', '0', '1']` (`1012`)):**

```
in:                  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [IDX] 0 7 [IDX] 1 8 [IDX] 2 9 [SEP] [IDX] 0 5 [IDX] 1 2 [SEP]
out (sum_first):     [IDX] 0 2 1  [IDX] 1 1 1  [IDX] 2 0 1  [IDX] 3 1 0  [EOS]
out (carry_first):   [IDX] 0 1 2  [IDX] 1 1 1  [IDX] 2 1 0  [IDX] 3 0 1  [EOS]
```

**Example 2 (`alphabet = ['q', '#', '7']` (`B = 3`: `q = 0`, `# = 1`, `7 = 2`), `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `output = ['q', '7']` (`6`)):**

```
in:                  [BOS] q # 7 [SEP] [IDX] q 7 [IDX] # # [SEP] [IDX] q # [SEP]
out (sum_first):     [IDX] q q #  [IDX] # 7 q  [EOS]
out (carry_first):   [IDX] q # q  [IDX] # q 7  [EOS]
```

## 3. Supervision & Rollout

Let `P` be the length of the addressed problem prefix ending at the third `[SEP]`, let `trace` be the concatenation of the `K + 1` column entries (`[IDX] addr(k) o_k c_{k+1}` or `[IDX] addr(k) c_{k+1} o_k` when `use_query_tokens=False`, and `[IDX] addr(k) [S] o_k [C] c_{k+1}` or `[IDX] addr(k) [C] c_{k+1} [S] o_k` when `use_query_tokens=True`), and let `targets = (*trace, EOS)`:

- **Teacher-forced sequence labels (`build_addressed_ar_labelled_example`)**:
  - `example` appends `trace` to the addressed problem prefix (length `P + len(trace)`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + len(trace)`), supervising every trace token (`[IDX]`, `addr(k)`, optional `[S]` / `[C]`, `o_k`, `c_{k+1}`) and the terminating `[EOS]` in a single forward pass via `SubspaceLMHead` + `next_token_cross_entropy`, while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_addressed_ar`)**: Free-running greedy generation starts after the third `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one until `[EOS]` (or the maximum well-formed trace length for `max(L_a, L_b) + 1` columns is reached), and parses each `[IDX] addr(k) ...` segment into column predictions.

## 4. Evaluation Metrics (`evaluate_addressed_ar`)

Given teacher-forced greedy predictions across all `len(targets)` supervised trace positions of each problem (where column `k` in `0 .. K` spans the entry `[IDX] addr(k) ...` with `(pred_o[k], pred_c[k])` unpacked from its two value positions according to `step_order`, and the final supervised position after column `K` predicts `[EOS]` while the position after each non-final column `k < K` predicts `[IDX]`), `evaluate_addressed_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where every supervised trace prediction (including all `[IDX]` markers, `addr(k)` tokens, optional `[S]` / `[C]` tokens, `o_k`, `c_{k+1}`, and the final `[EOS]`) matches `targets`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature `[EOS]` before the end of column `K`, `[IDX]` predicted after every column `k < K`, and `[EOS]` predicted immediately after column `K`.
