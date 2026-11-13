# Addition — Parallel Classification Variant (`cls`)

The `cls` variant (`experiments/addition/variants/cls/`) frames [the `addition` task](../task.md) as per-step parallel classification through `ParallelClassificationHead`, driven by a step-by-step harness that updates the incoming carry `c_k` and accumulated sum digits `o_0, ..., o_{k-1}` between columns.

At each column step `k` (`0 <= k <= K`), `ParallelClassificationHead.heads[0..2]` read the hidden state at the final position (`lengths - 1`) of the step-`k` sequence to predict `(o_k, c_{k+1}, flag_k)` in parallel:
- `heads[0]` predicts the sum digit `o_k`.
- `heads[1]` predicts the outgoing carry `c_{k+1}` (`s_0` or `s_1`).
- `heads[2]` predicts the halting flag `flag_k` (`FLAG_CONTINUE` for `k < K`, `FLAG_FINISHED` for `k == K`).

## 1. Sequence Layout

At column `k` (`0 <= k <= K`), the shared problem prefix ([`task.md` §2](../task.md#2-shared-problem-prefix--embedding-modes-experimentsadditioninputspy)) is followed by the incoming carry `c_k`, a `[SEP]` delimiter, and the sum digits `o_0, ..., o_{k-1}` emitted so far:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}
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
| Incoming carry | `c_k` | `TokenType.CARRY` (`4`) | `k` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output so far | `o_0, ..., o_{k-1}` | `TokenType.OUTPUT` (`5`) | `1, 2, ..., k` |

Position coupling at column `k`:
- Output `position_ids` start at `1` (`o_0` at `1`, `o_{k-1}` at `k`) while the fourth `[SEP]` sits at `position_id = 0`, and `c_k` in `TokenType.CARRY` is assigned `position_id = k`.
- Consequently, at every step `k` in `0 .. K`, the final query token (`[SEP]` at `k == 0`, `o_{k-1}` at `k >= 1`), the incoming carry `c_k` (`TokenType.CARRY`), and the operand digits `a_k` (`TokenType.OPERAND_A`) and `b_k` (`TokenType.OPERAND_B`) all share `position_id = k` with zero offset.

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `out = ['2', '1', '0', '1']` (`1012`)):**

```
k = 0:  in:  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 0 [SEP]
        out: ('2', '1', [FLAG_CONTINUE])

k = 1:  in:  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2
        out: ('1', '1', [FLAG_CONTINUE])

k = 2:  in:  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1
        out: ('0', '1', [FLAG_CONTINUE])

k = 3:  in:  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 0
        out: ('1', '0', [FLAG_FINISHED])
```

**Example 2 (`alphabet = ['q', '#', '7']`, `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `out = ['q', '7']` (`6`)):**

```
k = 0:  in:  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] q [SEP]
        out: ('q', '#', [FLAG_CONTINUE])

k = 1:  in:  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] # [SEP] q
        out: ('7', 'q', [FLAG_FINISHED])
```

## 3. Supervision & Rollout

Because the incoming carry slot `c_k` is updated in place from step to step (`c_0` at `k = 0`, `c_1` at `k = 1`, ...), each step `k` has a distinct input sequence rather than extending step `k - 1` as a prefix:

- **Teacher-forced step examples**: a problem with `K + 1` ground-truth column transitions yields `K + 1` per-step `LabelledExample`s (`k = 0 .. K`). At step `k`, `example` is built from the ground-truth incoming carry `c_k` and ground-truth prefix `output = (o_0, ..., o_{k-1})`, and `labels = (o_k, c_{k+1}, flag_k)` supervises `heads[0..2]` at the final token (`lengths - 1`) via `parallel_classification_cross_entropy`.
- **Rollout (`rollout_cls`)**: Free-running greedy execution starts at `k = 0` with `carry = s_0` and `output = ()`. At each step `k` (up to `max(L_a, L_b) + 1` steps):
  1. The harness builds the step-`k` `Example` with the current `carry` and `output` and runs one forward pass through `model`.
  2. It decodes `(o_k, c_{k+1}, flag_k) = (argmax(heads[0]), argmax(heads[1]), argmax(heads[2]))` at the final token (`lengths - 1`).
  3. If `flag_k == FLAG_FINISHED`, rollout stops and returns the emitted `(o_k, c_{k+1}, flag_k)` step tuples; otherwise the harness updates `carry = c_{k+1}`, appends `o_k` to `output`, and advances to `k + 1`.

## 4. Evaluation Metrics (`evaluate_cls`)

Given teacher-forced greedy predictions `(pred_o[k], pred_c[k], pred_flag[k])` from `heads[0..2]` across all `K + 1` steps of each evaluated problem (`k` in `0 .. K`):

- **`exact_match`** *(per problem)*: fraction of problems where every step `k` in `0 .. K` predicts `pred_o[k] == o_k`, `pred_c[k] == c_{k+1}`, and `pred_flag[k] == flag_k`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature halt (`pred_flag[k] == FLAG_CONTINUE` for every `k < K`) and halts on the final column (`pred_flag[K] == FLAG_FINISHED`).
