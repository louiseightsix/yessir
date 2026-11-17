# Addition — Full-Trace Autoregressive Variant (`ar`)

The `ar` variant (`experiments/addition/variants/ar/`) frames [the `addition` task](../task.md) as pure language-model full-trace autoregressive next-token prediction through `SubspaceLMHead` (`input_ids` only, with `token_type_ids=None` and `position_ids=None`).

Unlike [`cls`](cls.md) and [`step_ar`](step_ar.md)—where an external harness updates the carry slot, strips transient step tokens between columns, and optionally attaches section `token_type_ids` / `position_ids`—`ar` feeds the problem prompt once as raw `input_ids` and lets the model generate the entire multi-column trace into its own context until `[EOS]`, passing zero side-channel metadata during generation. Holding `SubspaceLMHead` and `input_ids`-only (`nope`) embeddings fixed, comparing `ar` against [`step_ar`](step_ar.md) (`--embedding-mode nope`) isolates free-running unrolled scratchpad execution against per-step external state rewriting.

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - **`StepOrder.SUM_FIRST` (`"sum_first"`, default)**: each column `k` emits `(o_k, c_{k+1})`, giving the full `2 * (K + 1) + 1` target sequence `(o_0, c_1, o_1, c_2, ..., o_K, c_{K+1}, [EOS])`.
   - **`StepOrder.CARRY_FIRST` (`"carry_first"`)**: each column `k` emits `(c_{k+1}, o_k)`, giving the full `2 * (K + 1) + 1` target sequence `(c_1, o_0, c_2, o_1, ..., c_{K+1}, o_K, [EOS])`.
2. **`use_query_tokens: bool`** (`False` default):
   - **`False`**: each column `k` emits `(o_k, c_{k+1})` (or `(c_{k+1}, o_k)`) directly (`2 * (K + 1) + 1` target tokens total).
   - **`True`**: each value token in the trace is preceded by its self-emitted query token (`[S]` for `o_k`, `[C]` for `c_{k+1}`), so each column `k` emits `([S], o_k, [C], c_{k+1})` under `sum_first` or `([C], c_{k+1}, [S], o_k)` under `carry_first`, followed by `[EOS]` (`4 * (K + 1) + 1` target tokens total).

In all settings, the outgoing carry `c_{k+1}` of column `k` remains in the trace, where subsequent columns can attend to it directly as the incoming carry for column `k + 1`, and halting is represented by emitting `[EOS]` immediately after completing column `K`.

## 1. Sequence Layout

Given the shared problem prefix `[BOS] alphabet [SEP] a [SEP] b [SEP]` of length `P = B + L_a + L_b + 4` (`token_type_ids=None`, `position_ids=None`), the trace after the third `[SEP]` appends each completed column `k` directly to `input_ids`:

```
sum_first   (use_query_tokens=False):  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  o_0 c_1  o_1 c_2 ... o_K c_{K+1}
carry_first (use_query_tokens=False):  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_1 o_0  c_2 o_1 ... c_{K+1} o_K
sum_first   (use_query_tokens=True):   [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  [S] o_0 [C] c_1  [S] o_1 [C] c_2 ... [S] o_K [C] c_{K+1}
carry_first (use_query_tokens=True):   [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  [C] c_1 [S] o_0  [C] c_2 [S] o_1 ... [C] c_{K+1} [S] o_K
```

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
- **Rollout (`rollout_ar`)**: Free-running greedy generation starts with an empty trace after the third `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` directly to `input_ids` one by one (up to `(4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1` tokens, with no side-channel metadata), and stops as soon as `[EOS]` is emitted.

## 4. Evaluation Metrics (`evaluate_ar`)

`evaluate_ar` runs batched KV-cached free-running greedy rollout from each problem's `[BOS] ... [SEP]` prompt until `[EOS]` or `(4 if use_query_tokens else 2) * (max(L_a, L_b) + 1) + 1` tokens, unpacks emitted columns into `StepPrediction`s (`pred_o[k]`, `pred_c[k]`, `syntax_ok`, `halt_ok`), and computes the rollout metrics, single-pass teacher-forced `oracle_step_accuracy`, and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems whose free-running rollout (including `[S]` and `[C]` when `use_query_tokens=True`, and the final `[EOS]`) equals `targets` (`(*trace, EOS)`).
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` reference column steps across all evaluated problems where column `k` was reached during rollout, `syntax_ok` holds (`[S]` and `[C]` match when `use_query_tokens=True`), and both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`oracle_step_accuracy`** *(per column step `k` in `0 .. K`)*: fraction of the `K + 1` reference column steps where `syntax_ok`, `pred_o[k] == o_k`, and `pred_c[k] == c_{k+1}` hold under a single-pass teacher-forced forward pass on the ground-truth trace.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` reference column steps where column `k` was reached with `syntax_ok` and `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of reference column steps with `c_k != s_0` or `c_{k+1} != s_0` where column `k` was reached with `syntax_ok` and both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the rollout halts at the exact right step—completing all `K + 1` columns without premature `[EOS]` and emitting `[EOS]` immediately after column `K`.

When a rollout emits `[EOS]` at a column boundary (`emitted[0] == EOS` or `emitted[stride * k] == EOS` for `1 <= k <= K`), generation stops at column `k - 1` with `halt_ok = False`, so a boundary premature halt is classified as `syntax_malformation` (after any earlier value error in columns `0 .. k - 1`) and unreached columns `k .. K` count as missed in `step_accuracy`; emitting `[EOS]` mid-column in a value slot (e.g. `emitted[2 * k + 1] == EOS` after a non-`[EOS]` `t_0` when `use_query_tokens=False`) is classified as `out_of_alphabet`.
