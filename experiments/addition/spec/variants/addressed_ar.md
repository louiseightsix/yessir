# Addition — Addressed Full-Trace Autoregressive Variant (`addressed_ar`)

The `addressed_ar` variant (`experiments/addition/variants/addressed_ar/`) frames [the `addition` task](../task.md) as full-trace autoregressive next-token prediction through `SubspaceLMHead`, where every operand digit and every trace step is paired with its **explicit column index written as a base-`B` numeral in the prompt's own alphabet**.

Why explicit in-sequence column addresses:
- In [`ar`](ar.md), aligning column `k` relies either on sinusoidal `position_ids` (`TOKEN_TYPE_PE`, which encounters unseen sinusoidal phases when operand lengths exceed the training range `L > L_train`) or on counting `k` tokens from `[SEP]` under causal masking (`TOKEN_TYPE_NOPE` / `NOPE`, where attention softmax weights dilute over long sequences).
- Writing each digit's column index `j` directly in the token stream in base `B` allows the model to increment the column counter `addr(k)` in its trace and retrieve `a_k` and `b_k` by content-based address matching (`addr(j) == addr(k)`), without sinusoidal coordinate drift or long-range token counting.

### Base-`B` column addresses (`addr(j, alphabet)`)

For any non-negative column index `j >= 0` and ordered alphabet `alphabet = [s_0, ..., s_{B-1}]` of base `B = len(alphabet)`:
- `addr(0, alphabet) = (s_0,)`.
- For `j > 0`, `addr(j, alphabet)` is the non-empty tuple of base-`B` digits of `j` in least-significant-first (`LSD -> MSD`) order:
  - For `0 <= j < B`, `addr(j, alphabet) = (s_j,)` (a single symbol).
  - For `B <= j < B^2`, `addr(j, alphabet) = (s_{j % B}, s_{j // B})` (e.g. column `24` in Base 10 has `addr(24) = ('4', '2')`; column `5` in Base 2 has `addr(5) = (s_1, s_0, s_1)`).

### Variant parameters

1. **`step_order: StepOrder`** (`StepOrder.SUM_FIRST` default, or `StepOrder.CARRY_FIRST`):
   - `SUM_FIRST` (`"sum_first"`): each column `k` emits `o_k` before `c_{k+1}`.
   - `CARRY_FIRST` (`"carry_first"`): each column `k` emits `c_{k+1}` before `o_k`.
2. **`use_query_tokens: bool`** (`False` default):
   - `False`: each column `k` in the trace emits `[STEP] addr(k) o_k c_{k+1}` (or `[STEP] addr(k) c_{k+1} o_k` under `carry_first`).
   - `True`: each value token in the trace is preceded by its explicit query trigger (`[S]` for `o_k`, `[C]` for `c_{k+1}`): `[STEP] addr(k) [S] o_k [C] c_{k+1}` under `sum_first`, or `[STEP] addr(k) [C] c_{k+1} [S] o_k` under `carry_first`.
3. **`include_alphabet: bool`** (`True` default):
   - `True`: includes the explicit `s_0 ... s_{B-1} [SEP]` alphabet section after `[BOS]`.
   - `False` *(zero-preamble mode)*: omits `s_0 ... s_{B-1} [SEP]` from the prompt and requires operand lengths `L_a, L_b >= B` so every symbol `s_0, ..., s_{B-1}` appears in ascending order as the single-digit column addresses `addr(0) .. addr(B - 1)` of the operands.

After column `K`, the trace terminates with `[EOS]`.

## 1. Sequence Layout

Major sections (`alphabet`, `a`, `b`, and the output `trace`) are separated by `[SEP]`, matching [`ar`](ar.md). Within `a`, `b`, and `trace`, each column entry begins with `[STEP]` (`SpecialToken.STEP`) followed by its base-`B` address `addr(j)` and value token(s), so multi-digit addresses (`j >= B`) are unambiguously delimited in all embedding modes:

```
Prefix:
  [BOS]  s_0 ... s_{B-1}
  [SEP]  [STEP] addr(0) a_0  [STEP] addr(1) a_1 ... [STEP] addr(L_a-1) a_{L_a-1}
  [SEP]  [STEP] addr(0) b_0  [STEP] addr(1) b_1 ... [STEP] addr(L_b-1) b_{L_b-1}
  [SEP]

Trace (sum_first):
  [STEP] addr(0) o_0 c_1  [STEP] addr(1) o_1 c_2 ... [STEP] addr(K) o_K c_{K+1}

Trace (carry_first):
  [STEP] addr(0) c_1 o_0  [STEP] addr(1) c_2 o_1 ... [STEP] addr(K) c_{K+1} o_K
```

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet *(if `include_alphabet=True`)* | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter *(if `include_alphabet=True`)* | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` entry `j` (`0 <= j < L_a`) | `[STEP]`, `addr(j)`, `a_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_A` (`2`) | `j` (for all tokens in entry `j`) |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` entry `j` (`0 <= j < L_b`) | `[STEP]`, `addr(j)`, `b_j` | `SPECIAL` (`0`), `ADDRESS` (`6`), `OPERAND_B` (`3`) | `j` (for all tokens in entry `j`) |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Trace column `k` marker (`0 <= k <= K`) | `[STEP]` | `TokenType.SPECIAL` (`0`) | `k` |
| Trace column `k` address | `addr(k)` | `TokenType.ADDRESS` (`6`) | `k` |
| Trace column `k` first token *(with optional `[S]` / `[C]`)* | `([S],) o_k` (`sum_first`) / `([C],) c_{k+1}` (`carry_first`) | `(SPECIAL (0),) OUTPUT` (`5`) / `(SPECIAL (0),) CARRY` (`4`) | `k` |
| Trace column `k` second token *(with optional `[C]` / `[S]`)* | `([C],) c_{k+1}` (`sum_first`) / `([S],) o_k` (`carry_first`) | `(SPECIAL (0),) CARRY` (`4`) / `(SPECIAL (0),) OUTPUT` (`5`) | `k + 1` |

## 2. Examples

**Example 1 (`alphabet = ['0'..'9']`, `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `out = ['2', '1', '0', '1']` (`1012`)):**

```
in:                  [BOS] 0 1 2 3 4 5 6 7 8 9 [SEP] [STEP] 0 7 [STEP] 1 8 [STEP] 2 9 [SEP] [STEP] 0 5 [STEP] 1 2 [SEP]
out (sum_first):     [STEP] 0 2 1  [STEP] 1 1 1  [STEP] 2 0 1  [STEP] 3 1 0  [EOS]
out (carry_first):   [STEP] 0 1 2  [STEP] 1 1 1  [STEP] 2 1 0  [STEP] 3 0 1  [EOS]
```

**Example 2 (`alphabet = ['q', '#', '7']` (`B = 3`: `q = 0`, `# = 1`, `7 = 2`), `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `out = ['q', '7']` (`6`)):**

```
in:                  [BOS] q # 7 [SEP] [STEP] q 7 [STEP] # # [SEP] [STEP] q # [SEP]
out (sum_first):     [STEP] q q #  [STEP] # 7 q  [EOS]
out (carry_first):   [STEP] q # q  [STEP] # q 7  [EOS]
```

## 3. Supervision & Rollout

Let `P` be the length of the addressed problem prefix ending at the third `[SEP]`, let `trace` be the concatenation of the `K + 1` column entries `[STEP] addr(k) o_k c_{k+1}` (or `[STEP] addr(k) c_{k+1} o_k` under `carry_first`), and let `targets = (*trace, EOS)`:

- **Teacher-forced sequence labels (`build_addressed_ar_labelled_example`)**:
  - `example` appends `trace` to the addressed problem prefix (length `P + len(trace)`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + len(trace)`), supervising every trace token (`[STEP]`, `addr(k)`, `o_k`, `c_{k+1}`) and the terminating `[EOS]` in a single forward pass via `SubspaceLMHead` + `next_token_cross_entropy`, while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_addressed_ar`)**: Free-running greedy generation starts after the third `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one until `[EOS]` (or the maximum well-formed trace length for `max(L_a, L_b) + 1` columns is reached), and parses each `[STEP] addr(k) ...` segment into column predictions.

## 4. Evaluation Metrics (`evaluate_addressed_ar`)

Given teacher-forced greedy predictions across all `len(targets)` supervised trace positions of each problem (where column `k` in `0 .. K` spans the entry `[STEP] addr(k) t_{k,0} t_{k,1}` with `(pred_o[k], pred_c[k])` unpacked from `(t_{k,0}, t_{k,1})` according to `step_order`, and the final supervised position after column `K` predicts `[EOS]` while the position after each non-final column `k < K` predicts `[STEP]`), `evaluate_addressed_ar` computes the five metrics and 4-way first-error taxonomy defined in [`task.md` §3](../task.md#3-evaluation-metrics-experimentsadditionevaluationpy):

- **`exact_match`** *(per problem)*: fraction of problems where every supervised trace prediction (including all `[STEP]` markers, `addr(k)` tokens, `o_k`, `c_{k+1}`, and the final `[EOS]`) matches `targets`.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `addr(k)`, `o_k`, and `c_{k+1}` all match the ground truth for column `k`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` column steps across all evaluated problems where `pred_o[k] == o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of column steps with `c_k != s_0` or `c_{k+1} != s_0` where `addr(k)`, `o_k`, and `c_{k+1}` all match the ground truth for column `k`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right column `K`—no premature `[EOS]` before the end of column `K`, `[STEP]` predicted after every column `k < K`, and `[EOS]` predicted immediately after column `K`.
