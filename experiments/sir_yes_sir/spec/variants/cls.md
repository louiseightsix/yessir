# Sir, Yes Sir! — Parallel Classification Variant (`cls`)

The `cls` variant (`experiments/sir_yes_sir/variants/cls/`) frames [the `sir_yes_sir` task](../task.md) as per-step parallel classification through `ParallelClassificationHead`.

At each step `k` (`0 <= k < L`) with previous outputs `o_0, ..., o_{k-1}`, `ParallelClassificationHead.heads[0..1]` read the hidden state at the query position for step `k` to predict `(o_k, flag_k)` in parallel:
- `heads[0]` predicts the successor symbol `o_k`.
- `heads[1]` predicts the halting flag `flag_k` (`FLAG_CONTINUE` for `k < L - 1`, `FLAG_FINISHED` for `k == L - 1`).

## 1. Examples

**Example 1 (`alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']` -> `o = ['2', '0', '1']`):**

```
k = 0:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP]
        out: ('2', [FLAG_CONTINUE])

k = 1:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2
        out: ('0', [FLAG_CONTINUE])

k = 2:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 0
        out: ('1', [FLAG_FINISHED])
```

**Example 2 (`alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']` -> `o = ['7', 'q', '#', 'q']`):**

```
k = 0:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP]
        out: ('7', [FLAG_CONTINUE])

k = 1:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7
        out: ('q', [FLAG_CONTINUE])

k = 2:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q
        out: ('#', [FLAG_CONTINUE])

k = 3:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q #
        out: ('q', [FLAG_FINISHED])
```

## 2. Supervision & Greedy Generation

Using the shared `[BOS] alphabet [SEP] x [SEP] output` sequence layout ([`task.md` §2](../task.md#2-shared-sequence-layout--embedding-modes-experimentssir_yes_sirinputspy)), let `P = B + L + 3` be the length of the prompt prefix `[BOS] alphabet [SEP] x [SEP]`:

- **Supervision (`ClsSupervision` / `supervision`)**:
  - **`single_pass` (default, `build_cls_labelled_example`)**: the model reads the full teacher-forced sequence `build_example(alphabet, x, output=(o_0, ..., o_{L-2}))` (`prefix + (o_0, ..., o_{L-2})`, length `P + L - 1`) in a single forward pass, and `heads[0..1]` read the hidden states at every query position `P - 1, ..., P + L - 2` (`lengths=None`):
    - At `k == 0`, the query sits at index `P - 1` (the second `[SEP]` at `position_id = 0`) and predicts `(o_0, flag_0)`.
    - At `k >= 1`, the query sits at index `P - 1 + k` (`o_{k-1}` at `position_id = k`) and predicts `(o_k, flag_k)`.
    Because the attention mask is causal, the hidden state at the query for step `k` sees the exact same prefix as the step-`k` input in the examples above, giving the same `L` predictions as `L` separate forward passes. The loss averages over all `L` steps of every problem, matching the one-pass supervision of [`ar`](ar.md).
  - **`all_steps` (`build_cls_step_labelled_example`)**: emits `L` consecutive single-step examples per problem (`k = 0 .. L - 1`), each reading the final unpadded position `lengths - 1` to predict `(o_k, flag_k)`. At fixed `batch_size`, each optimiser step supervises `batch_size` steps drawn from `~batch_size / L` distinct problems (with consecutive rows sharing `(alphabet, x)`), compared to `batch_size` distinct problems (`L * batch_size` supervised steps) in `single_pass` and `batch_size` independent problems (`batch_size` supervised steps) in `random_step`.
  - **`random_step` (`build_cls_step_labelled_example`)**: emits one randomly sampled single-step example `k` (`0 <= k < L`) per problem, reading `lengths - 1` to predict `(o_k, flag_k)` so every row in a mini-batch comes from an independent `(alphabet, x)`.
- **Greedy generation (`greedy_generate_cls`)**: Free-running greedy generation starts with `output = ()` at `k = 0`, reads `(o_k, flag_k) = (argmax(heads[0]), argmax(heads[1]))` at the final token, appends `o_k` to `output`, and stops when `flag_k == FLAG_FINISHED` (or after `L` steps).

## 3. Evaluation Metrics (`evaluate_cls`)

`evaluate_cls` runs batched KV-cached free-running greedy rollout from each problem's `[BOS] alphabet [SEP] x [SEP]` prompt, feeding predicted `o_k` tokens forward and stopping per problem as soon as `pred_flag[k] == FLAG_FINISHED` or `L` steps have been emitted (`emitted_steps`, matching `greedy_generate_cls`):

- **`exact_match`** *(per problem)*: fraction of problems where `emitted_steps` has length `L` and every step `k` in `0 .. L - 1` predicts both `pred_o[k] == o_k` and `pred_flag[k] == flag_k`.
- **`step_accuracy`** *(per input step `k` in `0 .. L - 1`, `L` steps per problem)*: fraction of the `L` reference steps across all evaluated problems where step `k` was reached (`k < len(emitted_steps)`) and `pred_o[k] == o_k`. Because rollout stops immediately on `FLAG_FINISHED`, post-halt steps `k + 1 .. L - 1` after a premature halt at step `k` receive zero credit; note that at the premature halt step `k` itself, `cls` still emits `(pred_o[k], FLAG_FINISHED)` on separate heads (stopping *after* step `k`), whereas emitting `[EOS]` at index `k` in [`ar`](ar.md) occupies the symbol slot and stops *before* `o_k`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the rollout halts at the exact right step—`len(emitted_steps) == L`, `pred_flag[k] == FLAG_CONTINUE` for every `k < L - 1`, and `pred_flag[L - 1] == FLAG_FINISHED`.
