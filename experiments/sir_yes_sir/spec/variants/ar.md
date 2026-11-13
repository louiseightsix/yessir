# Sir, Yes Sir! — Autoregressive Variant (`ar`)

The `ar` variant (`experiments/sir_yes_sir/variants/ar/`) frames [the `sir_yes_sir` task](../task.md) as autoregressive next-token prediction through `SubspaceLMHead`.

For an input `x` of length `L`, the complete ground-truth output sequence has length `L + 1`: the `L` successor symbols `o_0, ..., o_{L-1}` followed by `[EOS]`.

## 1. Examples

**Example 1 (`alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']` -> `o = ['2', '0', '1']`):**

```
in:   [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP]
out:  2 0 1 [EOS]
```

**Example 2 (`alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']` -> `o = ['7', 'q', '#', 'q']`):**

```
in:   [BOS] q # 7 [SEP] # 7 q 7 [SEP]
out:  7 q # q [EOS]
```

## 2. Supervision & Greedy Generation

Using the shared `[BOS] alphabet [SEP] x [SEP] output` sequence layout ([`task.md` §2](../task.md#2-shared-sequence-layout--embedding-modes-experimentssir_yes_sirinputspy)), let `P = B + L + 3` be the length of the prompt prefix `[BOS] alphabet [SEP] x [SEP]` and `targets = (o_0, ..., o_{L-1}, EOS)` be the ground-truth target sequence of length `L + 1`:

- **Teacher-forced per-position labels (`build_ar_labelled_example`)**:
  - `example` is `build_example(alphabet, x, output=(o_0, ..., o_{L-1}))`, so `example.input_ids` is `prefix + (o_0, ..., o_{L-1})` (length `P + L`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + L`), so `SubspaceLMHead` + `next_token_cross_entropy` supervise predictions in a single forward pass from index `P - 1` (the second `[SEP]` at `position_id = 0`, predicting `o_0`) through index `P + L - 1` (`o_{L-1}` at `position_id = L`, predicting `EOS`), while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Greedy generation (`greedy_generate_ar`)**: Free-running greedy generation starts with `output = ()` after the second `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one (up to `L + 1` tokens), and stops as soon as `EOS` is emitted.

## 3. Evaluation Metrics (`evaluate_ar`)

Given teacher-forced greedy predictions `preds[0 .. L]` at positions `P - 1 .. P + L - 1` (`L + 1` predictions per problem):

- **`exact_match`** *(per problem)*: fraction of problems where all `L + 1` predictions match `(o_0, ..., o_{L-1}, EOS)`.
- **`step_accuracy`** *(per input step `k` in `0 .. L - 1`, `L` steps per problem)*: fraction of the `L` symbol positions across all evaluated problems where `preds[k] == o_k`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right step—no premature halt (`preds[k] != EOS` for every `k < L`) and halts immediately after `o_{L-1}` (`preds[L] == EOS`).
