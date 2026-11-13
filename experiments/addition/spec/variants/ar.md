# Addition — Full-Trace Autoregressive Variant (`ar`)

The `ar` variant (`experiments/addition/variants/ar/`) frames [the `addition` task](../task.md) as full-trace autoregressive next-token prediction through `SubspaceLMHead`.

Unlike [`cls`](cls.md) and [`step_ar`](step_ar.md)—where an external harness updates the carry slot and strips transient step tokens between columns—`ar` feeds the problem prompt once and lets the model generate the entire multi-column trace into its own context until `[EOS]`. Holding `SubspaceLMHead` fixed, comparing `ar` against [`step_ar`](step_ar.md) isolates free-running unrolled scratchpad execution against per-step external state rewriting.

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - **`StepOrder.SUM_FIRST` (`"sum_first"`, default)**: each column `k` emits `(o_k, c_{k+1})`, giving the full `2 * (K + 1) + 1` target sequence `(o_0, c_1, o_1, c_2, ..., o_K, c_{K+1}, [EOS])`.
   - **`StepOrder.CARRY_FIRST` (`"carry_first"`)**: each column `k` emits `(c_{k+1}, o_k)`, giving the full `2 * (K + 1) + 1` target sequence `(c_1, o_0, c_2, o_1, ..., c_{K+1}, o_K, [EOS])`.
2. **`use_query_tokens: bool`** (`False` default):
   - **`False`**: each column `k` emits `(o_k, c_{k+1})` (or `(c_{k+1}, o_k)`) directly (`2 * (K + 1) + 1` target tokens total).
   - **`True`**: each value token in the trace is preceded by its explicit query trigger (`[S]` for `o_k`, `[C]` for `c_{k+1}`), so each column `k` emits `([S], o_k, [C], c_{k+1})` under `sum_first` or `([C], c_{k+1}, [S], o_k)` under `carry_first`, followed by `[EOS]` (`4 * (K + 1) + 1` target tokens total, where `[S]` and `[C]` sit at the same `position_id` as the value token they precede).

In all settings, the outgoing carry `c_{k+1}` of column `k` remains in the trace, where subsequent columns can attend to it directly as the incoming carry for column `k + 1`, and halting is represented by emitting `[EOS]` immediately after completing column `K`.

## 1. Sequence Layout

Given the shared problem prefix ([`task.md` §2](../task.md#2-shared-problem-prefix--embedding-modes-experimentsadditioninputspy)) of length `P = B + L_a + L_b + 4`, the trace after the third `[SEP]` appends 2 tokens `(t_{2k}, t_{2k+1})` per completed column `k`:

```
sum_first:    [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  o_0 c_1  o_1 c_2 ... o_K c_{K+1}
carry_first:  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_1 o_0  c_2 o_1 ... c_{K+1} o_K
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
| Column `k` first token (`t_{2k}`) | `o_k` (`sum_first`) / `c_{k+1}` (`carry_first`) | `TokenType.OUTPUT` (`5`) / `TokenType.CARRY` (`4`) | `k` |
| Column `k` second token (`t_{2k+1}`) | `c_{k+1}` (`sum_first`) / `o_k` (`carry_first`) | `TokenType.CARRY` (`4`) / `TokenType.OUTPUT` (`5`) | `k + 1` |

Position coupling across the trace:
- For every column `k` in `0 .. K`, the first token of column `k` (`t_{2k}`) sits at `position_id = k`, and the second token of column `k` (`t_{2k+1}`) sits at `position_id = k + 1` (while the third `[SEP]` before `k = 0` sits at `position_id = 0`).
- Consequently, under both `sum_first` and `carry_first`, **both** query tokens that predict column `k`'s two outputs—the token preceding column `k` (`[SEP]` at `k = 0`, `t_{2k-1}` at `k >= 1`) and column `k`'s first token (`t_{2k}`)—have `position_id = k`, matching `a_k` (`TokenType.OPERAND_A`) and `b_k` (`TokenType.OPERAND_B`) at `position_id = k` with zero offset.
- At the end of column `K`, `t_{2K+1}` sits at `position_id = K + 1` and predicts `[EOS]`.

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `out = ['2', '1', '0', '1']` (`1012`)):**

```
in:                  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP]
out (sum_first):     2 1  1 1  0 1  1 0  [EOS]
out (carry_first):   1 2  1 1  1 0  0 1  [EOS]
```

**Example 2 (`alphabet = ['q', '#', '7']`, `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `out = ['q', '7']` (`6`)):**

```
in:                  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP]
out (sum_first):     q #  7 q  [EOS]
out (carry_first):   # q  q 7  [EOS]
```

## 3. Supervision & Rollout

Let `P = B + L_a + L_b + 4` be the length of the problem prefix `[BOS] alphabet [SEP] a [SEP] b [SEP]`, let `trace = (t_0, t_1, ..., t_{2K}, t_{2K+1})` be the `2 * (K + 1)` column tokens ordered by `step_order`, and let `targets = (*trace, EOS)` (length `2 * (K + 1) + 1`):

- **Teacher-forced sequence labels (`build_ar_labelled_example`)**:
  - `example` appends `trace` to the problem prefix (length `P + 2 * (K + 1)`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + 2 * (K + 1)`), so `SubspaceLMHead` + `next_token_cross_entropy` supervise all `2 * (K + 1) + 1` predictions of the problem in a single forward pass from index `P - 1` (the third `[SEP]`, predicting `t_0`) through index `P + 2 * K + 1` (`t_{2K+1}`, predicting `[EOS]`), while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_ar`)**: Free-running greedy generation starts with an empty trace after the third `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one (up to `2 * (max(L_a, L_b) + 1) + 1` tokens, assigning each trace slot `i` its deterministic `token_type_id` and `position_id` from `step_order`), and stops as soon as `[EOS]` is emitted.

## 4. Evaluation Metrics (`evaluate_ar`)

Given teacher-forced greedy predictions `preds[0 .. 2K + 2]` at the `2 * (K + 1) + 1` supervised positions of each problem (where column `k` in `0 .. K` corresponds to `(pred_o[k], pred_c[k])` unpacked from `(preds[2k], preds[2k + 1])` according to `step_order`, and `preds[2K + 2]` is the halting prediction after column `K`), `evaluate_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where all `2 * (K + 1) + 1` predictions match `targets` (`(*trace, EOS)`).
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right step—no premature `[EOS]` at any trace position `0 .. 2K + 1`, and `preds[2K + 2] == EOS` immediately after column `K`.
