# Addition — Full-Trace Autoregressive Variant (`ar`)

The `ar` variant (`experiments/addition/variants/ar/`) frames [the `addition` task](../task.md) as full-trace autoregressive next-token prediction through `SubspaceLMHead`.

Unlike [`cls`](cls.md) and [`step_ar`](step_ar.md)—where an external harness updates the carry slot and strips transient step tokens between columns—`ar` feeds the problem prompt once and lets the model generate the entire multi-column trace into its own context until `[EOS]`. Holding `SubspaceLMHead` fixed, comparing `ar` against [`step_ar`](step_ar.md) isolates free-running unrolled scratchpad execution against per-step external state rewriting.

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - **`StepOrder.SUM_FIRST` (`"sum_first"`, default)**: each column `k` emits `(o_k, c_{k+1})`, giving the full `2 * (K + 1) + 1` target sequence `(o_0, c_1, o_1, c_2, ..., o_K, c_{K+1}, [EOS])`.
   - **`StepOrder.CARRY_FIRST` (`"carry_first"`)**: each column `k` emits `(c_{k+1}, o_k)`, giving the full `2 * (K + 1) + 1` target sequence `(c_1, o_0, c_2, o_1, ..., c_{K+1}, o_K, [EOS])`.
2. **`use_query_tokens: bool`** (`False` default):
   - **`False`**: each column `k` emits `(o_k, c_{k+1})` (or `(c_{k+1}, o_k)`) directly (`2 * (K + 1) + 1` target tokens total).
   - **`True`**: each value token in the trace is preceded by its query token (`[S]` for `o_k`, `[C]` for `c_{k+1}`), so each column `k` emits `([S], o_k, [C], c_{k+1})` under `sum_first` or `([C], c_{k+1}, [S], o_k)` under `carry_first`, followed by `[EOS]` (`4 * (K + 1) + 1` target tokens total, where `[S]` and `[C]` for column `k` both sit at `(TokenType.SPECIAL, k)`).

In all settings, the outgoing carry `c_{k+1}` of column `k` remains in the trace, where subsequent columns can attend to it directly as the incoming carry for column `k + 1`, and halting is represented by emitting `[EOS]` immediately after completing column `K`.

## 1. Sequence Layout

Given the shared problem prefix ([`task.md` §2](../task.md#2-shared-problem-prefix--embedding-modes-experimentsadditioninputspy)) of length `P = B + L_a + L_b + 4`, the trace after the third `[SEP]` appends each completed column `k` (`2` tokens `(o_k, c_{k+1})` / `(c_{k+1}, o_k)` when `use_query_tokens=False`, or `4` tokens `([S], o_k, [C], c_{k+1})` / `([C], c_{k+1}, [S], o_k)` when `use_query_tokens=True`):

```
sum_first   (use_query_tokens=False):  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  o_0 c_1  o_1 c_2 ... o_K c_{K+1}
carry_first (use_query_tokens=False):  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_1 o_0  c_2 o_1 ... c_{K+1} o_K
sum_first   (use_query_tokens=True):   [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  [S] o_0 [C] c_1  [S] o_1 [C] c_2 ... [S] o_K [C] c_{K+1}
carry_first (use_query_tokens=True):   [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  [C] c_1 [S] o_0  [C] c_2 [S] o_1 ... [C] c_{K+1} [S] o_K
```

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` | `a_0, ..., a_{L_a-1}` | `TokenType.OPERAND_A` (`2`) | `0, 1, ..., L_a - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` | `b_0, ..., b_{L_b-1}` | `TokenType.OPERAND_B` (`3`) | `0, 1, ..., L_b - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Column `k` query tokens *(if `use_query_tokens=True`)* | `[S]`, `[C]` | `TokenType.SPECIAL` (`0`) | `k` |
| Column `k` sum digit | `o_k` | `TokenType.OUTPUT` (`5`) | `k` |
| Column `k` outgoing carry | `c_{k+1}` | `TokenType.CARRY` (`4`) | `k + 1` |

*(Under `carry_first`, column `k` emits `c_{k+1}` before `o_k`—and `[C] c_{k+1}` before `[S] o_k` when `use_query_tokens=True`—keeping the exact same `token_type_ids` and `position_ids` as in [`step_ar`](step_ar.md).)*

Position coupling across the trace:
- For every column `k` in `0 .. K`, `o_k` sits at `(TokenType.OUTPUT, k)` and `c_{k+1}` sits at `(TokenType.CARRY, k + 1)` (the index of the next column `k + 1` that `c_{k+1}` feeds into), while any query tokens `[S]` and `[C]` for column `k` sit at `(TokenType.SPECIAL, k)`.
- Consequently, at column `k + 1`, its incoming carry `c_{k+1}` in the trace always sits at `(TokenType.CARRY, k + 1)`—matching `a_{k+1}` (`TokenType.OPERAND_A`) and `b_{k+1}` (`TokenType.OPERAND_B`) at `position_id = k + 1` with zero offset under both `sum_first` and `carry_first`.

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `output = ['2', '1', '0', '1']` (`1012`)):**

```
in:                  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP]
out (sum_first):     2 1  1 1  0 1  1 0  [EOS]
out (carry_first):   1 2  1 1  1 0  0 1  [EOS]
```

**Example 2 (`alphabet = ['q', '#', '7']`, `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `output = ['q', '7']` (`6`)):**

```
in:                  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP]
out (sum_first):     q #  7 q  [EOS]
out (carry_first):   # q  q 7  [EOS]
```

## 3. Supervision & Rollout

Let `P = B + L_a + L_b + 4` be the length of the problem prefix `[BOS] alphabet [SEP] a [SEP] b [SEP]`, let `trace` be the column tokens across `k = 0 .. K` ordered by `step_order` (`2 * (K + 1)` tokens when `use_query_tokens=False`, or `4 * (K + 1)` tokens including `[S]` and `[C]` when `use_query_tokens=True`), and let `targets = (*trace, EOS)` (length `len(trace) + 1`):

- **Teacher-forced sequence labels (`build_ar_labelled_example`)**:
  - `example` appends `trace` to the problem prefix (length `P + len(trace)`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + len(trace)`), so `SubspaceLMHead` + `next_token_cross_entropy` supervise all `len(trace) + 1` predictions of the problem (including `[S]` and `[C]` when `use_query_tokens=True`) in a single forward pass from index `P - 1` (the third `[SEP]`, predicting `trace[0]`) through index `P + len(trace) - 1` (`trace[-1]`, predicting `[EOS]`), while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_ar`)**: Free-running greedy generation starts with an empty trace after the third `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one (up to `(4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1` tokens, assigning each trace slot `i` its deterministic `token_type_id` and `position_id` from `step_order` and `use_query_tokens`), and stops as soon as `[EOS]` is emitted.

## 4. Evaluation Metrics (`evaluate_ar`)

Given teacher-forced greedy predictions `preds` across all `len(targets)` supervised positions of each problem (where column `k` in `0 .. K` spans the `2` or `4` trace slots of column `k` with `(pred_o[k], pred_c[k])` unpacked from its two value positions according to `step_order`, and `preds[-1]` is the halting prediction after column `K`), `evaluate_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where all `len(targets)` predictions (including `[S]` and `[C]` when `use_query_tokens=True`, and the final `[EOS]`) match `targets` (`(*trace, EOS)`).
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right step—no premature `[EOS]` at any trace position before `preds[-1]`, and `preds[-1] == EOS` immediately after column `K`.
