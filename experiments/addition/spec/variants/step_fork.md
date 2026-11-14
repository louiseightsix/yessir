# Addition — Per-Step Branching Query Variant (`step_fork`)

The `step_fork` variant (`experiments/addition/variants/step_fork/`) frames [the `addition` task](../task.md) as three independent single-token queries per column read through `SubspaceLMHead`, driven by the same step-by-step harness as [`cls`](cls.md) and [`step_ar`](step_ar.md).

Why `step_fork` bridges [`cls`](cls.md) and [`step_ar`](step_ar.md):
- Comparing [`cls`](cls.md) directly against [`step_ar`](step_ar.md) changes two things at once: the output head (`ParallelClassificationHead` vs. `SubspaceLMHead`) and the within-step factorisation (in `cls`, `o_k`, `c_{k+1}`, and `flag_k` are predicted conditionally independently in parallel; in `step_ar`, the second and third predictions condition autoregressively on the first).
- `step_fork` isolates those two effects:
  - **Against [`cls`](cls.md)**: `step_fork` keeps the exact same order-free factorisation (`o_k`, `c_{k+1}`, and `flag_k` are predicted independently from the step-`k` state without conditioning on one another), changing only the readout mechanism from learned linear heads (`ParallelClassificationHead`) to query-token next-token prediction (`SubspaceLMHead`).
  - **Against [`step_ar`](step_ar.md)**: `step_fork` keeps `SubspaceLMHead`, isolating whether within-step autoregressive conditioning (`o_k -> c_{k+1}` or `c_{k+1} -> o_k`) helps compared to predicting `o_k` and `c_{k+1}` independently.

## 1. Sequence Layout & Query Tokens

At column `k` (`0 <= k <= K`), the base step prompt contains the shared problem prefix ([`task.md` §2](../task.md#2-shared-problem-prefix--embedding-modes-experimentsadditioninputspy)), the incoming carry `c_k`, and the sum digits `o_0, ..., o_{k-1}` emitted by earlier columns. From that shared prefix, `step_fork` forms three parallel query sequences by appending one of the three experiment-specific query tokens (`AdditionToken` in `vocab.EXPERIMENT_SPECIAL_RANGE`, `0xF8`–`0xFA`):

- **`[S]` (`AdditionToken.QUERY_SUM = 0xF8`)**: queries the sum digit `o_k`.
- **`[C]` (`AdditionToken.QUERY_CARRY = 0xF9`)**: queries the outgoing carry `c_{k+1}`.
- **`[F]` (`AdditionToken.QUERY_FLAG = 0xFA`)**: queries the halting flag `flag_k` (`FLAG_CONTINUE` or `FLAG_FINISHED`).

```
sum branch:    [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [S]
carry branch:  [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [C]
flag branch:   [BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  c_k  [SEP]  o_0 ... o_{k-1}  [F]
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
| Output history | `o_0, ..., o_{k-1}` | `TokenType.OUTPUT` (`5`) | `0, 1, ..., k - 1` |
| Query token | `[S]` / `[C]` / `[F]` | `TokenType.SPECIAL` (`0`) | `k` |

Position coupling at column `k`:
- Each historical sum digit `o_j` (`0 <= j < k`) sits at `position_id = j`.
- The trailing query token (`[S]`, `[C]`, or `[F]`) sits at `position_id = k`, matching `a_k` (`TokenType.OPERAND_A`), `b_k` (`TokenType.OPERAND_B`), and `c_k` (`TokenType.CARRY`) at `position_id = k` with zero offset.

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `output = ['2', '1', '0', '1']` (`1012`)):**

```
k = 0:  in ([S]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 0 [SEP] [S]  ->  out: 2
        in ([C]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 0 [SEP] [C]  ->  out: 1
        in ([F]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 0 [SEP] [F]  ->  out: [FLAG_CONTINUE]

k = 1:  in ([S]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 [S]  ->  out: 1
        in ([C]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 [C]  ->  out: 1
        in ([F]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 [F]  ->  out: [FLAG_CONTINUE]

k = 2:  in ([S]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 [S]  ->  out: 0
        in ([C]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 [C]  ->  out: 1
        in ([F]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 [F]  ->  out: [FLAG_CONTINUE]

k = 3:  in ([S]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 0 [S]  ->  out: 1
        in ([C]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 0 [C]  ->  out: 0
        in ([F]):  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] 7 8 9 [SEP] 5 2 [SEP] 1 [SEP] 2 1 0 [F]  ->  out: [FLAG_FINISHED]
```

**Example 2 (`alphabet = ['q', '#', '7']`, `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `output = ['q', '7']` (`6`)):**

```
k = 0:  in ([S]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] q [SEP] [S]    ->  out: q
        in ([C]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] q [SEP] [C]    ->  out: #
        in ([F]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] q [SEP] [F]    ->  out: [FLAG_CONTINUE]

k = 1:  in ([S]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] # [SEP] q [S]  ->  out: 7
        in ([C]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] # [SEP] q [C]  ->  out: q
        in ([F]):  [BOS] q # 7 [SEP] 7 # [SEP] # [SEP] # [SEP] q [F]  ->  out: [FLAG_FINISHED]
```

## 3. Supervision & Rollout

- **Teacher-forced step examples**: at column `k` (`0 <= k <= K`), let `P_k` be the length of each branched query sequence ending in `[S]`, `[C]`, or `[F]`. Each column transition yields 3 `LabelledExample`s:
  - Sum query (`[S]`): `labels = (IGNORE_INDEX,) * (P_k - 1) + (o_k,)`
  - Carry query (`[C]`): `labels = (IGNORE_INDEX,) * (P_k - 1) + (c_{k+1},)`
  - Flag query (`[F]`): `labels = (IGNORE_INDEX,) * (P_k - 1) + (flag_k,)`
  Each branch supervises only the final query position (`P_k - 1`) via `SubspaceLMHead` + `next_token_cross_entropy`. During training, each column step is drawn by generating a fresh problem and sampling a single column index `k` with terminal-step oversampling `p_terminal` ([`task.md` §1](../task.md#alphabets-problem-sampling--generalisation-yessiralphabet)), and a batch of `batch_size` column steps collates all `3 * batch_size` branched sequences (`[S]`, `[C]`, `[F]` per step) so every training batch supervises `batch_size` transitions and `3 * batch_size` target tokens (matching [`cls`](cls.md) and [`step_ar`](step_ar.md)); during evaluation, all `K + 1` steps of each problem are evaluated.
- **Rollout (`rollout_step_fork`)**: Free-running greedy execution starts at `k = 0` with `carry = s_0` and `output = ()`. At each step `k` (up to `max(L_a, L_b) + 1` steps):
  1. The harness collates the 3 branched queries (`[S]`, `[C]`, `[F]`) into a 3-row batch, runs one batched forward pass through the model, and takes the greedy `argmax` of `SubspaceLMHead` at the final token (`-1`) of each row to obtain `(o_k, c_{k+1}, flag_k)`.
  2. If `flag_k == FLAG_FINISHED`, rollout stops and returns the emitted `(o_k, c_{k+1}, flag_k)` step tuples; otherwise the harness updates `carry = c_{k+1}`, appends `o_k` to `output`, and advances to `k + 1`.

## 4. Evaluation Metrics (`evaluate_step_fork`)

Given teacher-forced greedy predictions `(pred_o[k], pred_c[k], pred_flag[k])` from the `[S]`, `[C]`, and `[F]` query branches at each step `k` in `0 .. K` across all evaluated problems, `evaluate_step_fork` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where every step `k` in `0 .. K` predicts `pred_o[k] == o_k`, `pred_c[k] == c_{k+1}`, and `pred_flag[k] == flag_k`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where both `pred_o[k] == o_k` and `pred_c[k] == c_{k+1}`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature halt (`pred_flag[k] == FLAG_CONTINUE` for every `k < K`) and halts on the final column (`pred_flag[K] == FLAG_FINISHED`).
