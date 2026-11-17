# Addition — Task Specification

This experiment tests whether a transformer trained from scratch can learn multi-step column addition conditioned on an **in-context numeral system** (an ordered alphabet of digit symbols supplied in the prompt) rather than memorising fixed symbol values or a fixed base in its weights, and whether the learned procedure transfers zero-shot to unseen bases, alphabets, and operand lengths.

It is implemented in six variants that share the task definition, embedding modes, and evaluation metrics below:
- [`variants/cls.md`](variants/cls.md): parallel per-step classification (`ParallelClassificationHead`) with per-step harness state updates.
- [`variants/step_fork.md`](variants/step_fork.md): three parallel single-token query branches (`[S]`, `[C]`, `[F]`) per step (`SubspaceLMHead`) with per-step harness state updates.
- [`variants/step_ar.md`](variants/step_ar.md): sequential per-step next-token prediction (`SubspaceLMHead`) with per-step harness state updates.
- [`variants/ar.md`](variants/ar.md): full-trace autoregressive next-token prediction (`SubspaceLMHead`) with no external state rewriting.
- [`variants/addressed_step_ar.md`](variants/addressed_step_ar.md): sequential per-step next-token prediction (`SubspaceLMHead`) with explicit base-`B` column addresses and per-step harness state updates (including optional zero-preamble mode `include_alphabet=False`).
- [`variants/addressed_ar.md`](variants/addressed_ar.md): full-trace autoregressive next-token prediction (`SubspaceLMHead`) with explicit base-`B` column addresses in the sequence (including optional zero-preamble mode `include_alphabet=False`).

## 1. Task: In-Context Column Addition (`experiments/addition/reference.py`)

Each problem instance supplies an ordered alphabet in the prompt (an in-context numeral system, not few-shot demonstrations) together with two operands, and asks for their base-`B` sum step by step from least-significant to most-significant column:

1. **Alphabet** (`alphabet = [s_0, s_1, ..., s_{B-1}]`): an ordered list of `B >= 2` distinct byte symbols drawn from the 240-byte symbol pool (`0x00`–`0xEF`; bytes `0xF0`–`0xFF` are reserved for special tokens so all symbols and control markers fit in a single 256-token byte vocabulary). The 0-indexed position `i` of a symbol `s_i` in `alphabet` defines its digit value `val(s_i) = i`, and the length `B = len(alphabet)` defines the base (radix). In particular, `s_0` represents `0` and `s_1` represents `1`.
2. **Operands** (`a = [a_0, a_1, ..., a_{L_a-1}]`, `b = [b_0, b_1, ..., b_{L_b-1}]`): non-empty sequences of `L_a >= 1` and `L_b >= 1` symbols drawn from `alphabet`, written **least-significant digit first (`LSD -> MSD`)**. Writing operands `LSD -> MSD` places column `k` at offset `k` from the start of each operand regardless of total operand length (whereas `MSD -> LSD` order would require measuring each operand's length before aligning column `0`). Operands are unpadded (`L_a` and `L_b` need not be equal): for `k >= L_a`, the missing digit `a_k` has value `0` (and likewise `b_k` has value `0` for `k >= L_b`). Leading zeros at the most-significant end (`s_0`) are permitted.

At each column step `k` (`0 <= k <= K`) with incoming carry symbol `c_k` (`c_0 = s_0` at `k = 0`, and `c_k in {s_0, s_1}` at every column for two-operand addition), let:
- `v_a = val(a_k)` if `k < L_a` else `0`
- `v_b = val(b_k)` if `k < L_b` else `0`
- `v_c = val(c_k)` (`0` if `c_k == s_0`, `1` if `c_k == s_1`)
- `total = v_a + v_b + v_c` (`0 <= total <= 2 * B - 1`)

The ground-truth column transition (`reference_step`) defines:
- **Sum digit (`o_k`)**: `alphabet[total % B]` (`s_{total}` if `total < B`, or `s_{total - B}` if `total >= B`).
- **Outgoing carry (`c_{k+1}`)**: `alphabet[total // B]` (`s_0` if `total < B`, or `s_1` if `total >= B`).
- **Halting (`flag_k`)**: `FLAG_FINISHED` if `k >= max(L_a, L_b) - 1` and `c_{k+1} == s_0`, else `FLAG_CONTINUE`.

Let `N = max(L_a, L_b)` be the longer operand length. A problem executes `K + 1` column steps (`k = 0 .. K`), producing the `LSD -> MSD` sum `output = [o_0, ..., o_K]`:
- **No final carry overflow (`c_N == s_0`)**: column `N - 1` emits `c_N = s_0` and `flag_{N-1} = FLAG_FINISHED`, so `K = N - 1` (`N` steps total).
- **Final carry overflow (`c_N == s_1`)**: column `N - 1` emits `c_N = s_1` and `flag_{N-1} = FLAG_CONTINUE`; one additional column runs at `K = N` with `v_a = 0`, `v_b = 0`, and `v_c = 1`, emitting `o_K = s_1`, `c_{K+1} = s_0`, and `flag_K = FLAG_FINISHED` (`N + 1` steps total).

Solving one column transition in context decomposes into four sub-operations:
- **Column indexing & missing-digit handling**: retrieving `a_k` from `a` (or substituting `0` when `k >= L_a`), `b_k` from `b` (or substituting `0` when `k >= L_b`), and the incoming carry `c_k` (`s_0` at `k = 0`, or the previous step's `c_k` at `k >= 1`).
- **In-context value lookup & base determination**: locating `a_k`, `b_k`, and `c_k` in the prompt's `alphabet` section to recover their numerical ranks `0 .. B - 1` and the base `B = len(alphabet)`.
- **Carry threshold & modular sum**: computing `total = v_a + v_b + v_c`, testing the threshold `total >= B` for `c_{k+1}`, and indexing `alphabet[total % B]` to emit `o_k`.
- **Halting**: detecting that both operands have been consumed (`k >= max(L_a, L_b) - 1`) and the outgoing carry is zero (`c_{k+1} == s_0`).

### Examples

Writing columns left-to-right from least-significant (`k = 0`) to most-significant (`k = K`, with `[C] = FLAG_CONTINUE` and `[F] = FLAG_FINISHED`):

**Example 1 (`alphabet = ['0'..'9']` (`B = 10`), `a = ['7', '8', '9']` (`987`), `b = ['5', '2']` (`25`) -> `output = ['2', '1', '0', '1']` (`1012`)):**

```
  column k:    0    1    2    3   (LSD -> MSD)
  carry c_k:   0    1    1    1
  operand a:   7    8    9    -
+ operand b:   5    2    -    -
-------------------------------
  sum   o_k:   2    1    0    1
  carry out:   1    1    1    0
  halt flag:  [C]  [C]  [C]  [F]
```

**Example 2 (`alphabet = ['q', '#', '7']` (`B = 3`: `q = 0`, `# = 1`, `7 = 2`), `a = ['7', '#']` (`5`), `b = ['#']` (`1`) -> `output = ['q', '7']` (`6`)):**

```
  column k:    0    1   (LSD -> MSD)
  carry c_k:   q    #
  operand a:   7    #
+ operand b:   #    -
---------------------
  sum   o_k:   q    7
  carry out:   #    q
  halt flag:  [C]  [F]
```

### Alphabets, Problem Sampling & Generalisation (`yessir.alphabet`)

As in [`sir_yes_sir`](../../sir_yes_sir/spec/task.md#alphabets-yessiralphabet), alphabets are resampled per problem with `create_alphabet(symbols, *, sample_size=B, rng=rng)` so digit values must be read in context:
- **Random symbols**: `create_alphabet(SymbolSet.ALL, sample_size=B, rng=rng)` — `B` distinct bytes sampled in random order from the 240-byte symbol pool (`0x00`–`0xEF`).
- **Permuted canonical symbols**: `create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)` (or `HEX[:B]`) — conventional numerals shuffled into a random order per problem (redrawn if the shuffle returns the ascending order, so a permuted alphabet always differs from the canonical one; in base `B = 2` it is always `['1', '0']`), testing whether a model exposed to canonical numerals overrides a familiar symbol-order prior in favour of the prompt's alphabet.

Problem and step sampling expose three distribution controls:
- **Prior mixture (`p_prior` in `0.0..1.0`, default `0.0`)**: each training problem uses the conventional canonical alphabet (`DECIMAL[:B]` / `HEX[:B]` in ascending order) with probability `p_prior`, and a random alphabet from `SymbolSet.ALL` with probability `1 - p_prior`. Because `SymbolSet.ALL` already includes the ASCII numeral bytes, `p_prior = 0.0` exposes the model to those byte embeddings without inducing a canonical ordering preference, while increasing `p_prior` strengthens the canonical symbol-order prior.
- **Carry-chain injection (`p_chain` in `0.0..1.0`, default `0.25`)**: under uniform random digit sampling in base `B`, a carry-propagating column (`v_a + v_b == B - 1`) occurs with probability `1 / B`, so a carry chain of length `m` occurs only at rate `~(1 / B)^m`. With probability `p_chain` (when `min(L_a, L_b) >= 2`), the problem generator injects a carry chain of random length `m` (`1 <= m < min(L_a, L_b)`), setting the starting column `i` so `v_a + v_b >= B` (carry generation) and the next `m` columns `j` so `v_a + v_b == B - 1` (carry propagation).
- **One step per problem & terminal-step oversampling (`p_terminal` in `0.0..1.0`, default `0.2`)**: in the step-by-step harnessed variants ([`cls`](variants/cls.md), [`step_fork`](variants/step_fork.md), [`step_ar`](variants/step_ar.md), [`addressed_step_ar`](variants/addressed_step_ar.md)), each column step `k` is a separate model forward pass, and only `1` of the `K + 1` columns has `flag_k = FLAG_FINISHED`. During training, each example is drawn by generating a fresh problem and sampling a single column index `k`—selecting the terminal step `k = K` with probability `p_terminal` and a uniform step `k in 0 .. K` otherwise—so every example in a training batch has an independent alphabet and operands while keeping `FLAG_FINISHED` well-represented.

Varying the base `B` and operand lengths `L_a, L_b` between training and evaluation probes two axes of generalisation:
- **Base generalisation**: training on a single base (e.g. `{10}`) allows hardcoding a fixed carry threshold `B = 10`, and training on two bases (e.g. `{2, 10}`) allows switching between two memorised routines without computing `B = len(alphabet)`. Training across multiple bases (e.g. `{2, 4, 8, 10}`) and evaluating zero-shot on unseen bases—such as `B = 3` (interpolation) and `B = 12` (extrapolation)—tests whether the model computes `B` and digit values from `alphabet`.
- **Length generalisation**: training on operand lengths `L_a, L_b` drawn from `1 .. L_train` (e.g. `1 .. 20`) and evaluating on longer operands (e.g. `21 .. 40`) tests whether column alignment, carry propagation, and halting extrapolate beyond the training length range.
- **Train/eval overlap in restricted regimes**: under default settings (`SymbolSet.ALL` 240-byte pool, `bases="2,4,8,10"`, `L = 1..20`, `p_prior = 0.0`), the problem universe is large enough that accidental overlap between training and the `num_eval_problems = 200` `id` split is negligible (`0 / 200`). Restricting the alphabet or length space—for instance `p_prior = 1.0` on short binary/quaternary problems, or a small `symbol_set="decimal"` pool paired with tiny `max_length` (e.g. `max_length = 3`)—shrinks the universe on short problems so `id` evaluates partly on seen training instances.

## 2. Shared Problem Prefix & Embedding Modes (`experiments/addition/inputs.py`)

[`cls`](variants/cls.md), [`step_fork`](variants/step_fork.md), [`step_ar`](variants/step_ar.md), and [`ar`](variants/ar.md) share the unaddressed problem prefix layout (`[BOS] alphabet [SEP] a [SEP] b [SEP]`; [`addressed_step_ar`](variants/addressed_step_ar.md) and [`addressed_ar`](variants/addressed_ar.md) use the same `[BOS] ... [SEP] ... [SEP] ... [SEP]` sections with explicit base-`B` column addresses attached to each digit):

```
[BOS]  s_0 ... s_{B-1}  [SEP]  a_0 ... a_{L_a-1}  [SEP]  b_0 ... b_{L_b-1}  [SEP]  ...
```

Each position has a token in `input_ids`, and (depending on `EmbeddingMode`) optional entries in `token_type_ids` and `position_ids`. Token type ids mark which section a token belongs to:

| Prefix section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `a` | `a_0, ..., a_{L_a-1}` | `TokenType.OPERAND_A` (`2`) | `0, 1, ..., L_a - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Operand `b` | `b_0, ..., b_{L_b-1}` | `TokenType.OPERAND_B` (`3`) | `0, 1, ..., L_b - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |

In the prefix, digit `a_k` in `TokenType.OPERAND_A` and digit `b_k` in `TokenType.OPERAND_B` both sit at `position_id = k` (`0`-indexed from least-significant to most-significant column).

Across all six variants, `experiments/addition/inputs.py` defines seven section types (`TokenType` in `vocab.TOKEN_TYPE_RANGE`, `0..6`) and four experiment-specific special tokens (`AdditionToken` in `vocab.EXPERIMENT_SPECIAL_RANGE`, `0xF8`–`0xFB`, alongside the shared `SpecialToken`s `[SEP] = 0xF0`, `[PAD] = 0xF1`, `[BOS] = 0xF2`, `[EOS] = 0xF3`, `[FLAG_CONTINUE] = 0xF5`, `[FLAG_FINISHED] = 0xF6`):

| `TokenType` | Value | Role | `position_id` at column `k` |
| :--- | :--- | :--- | :--- |
| `TokenType.SPECIAL` | `0` | Delimiters (`[BOS]`, `[SEP]`, `[EOS]`, `[IDX]`) and query tokens (`[S]`, `[C]`, `[F]`) | `0` in prefix; `k` for column-`k` markers/queries |
| `TokenType.ALPHABET` | `1` | Alphabet symbols `s_0, ..., s_{B-1}` | `0, 1, ..., B - 1` |
| `TokenType.OPERAND_A` | `2` | Operand `a` digits `a_0, ..., a_{L_a-1}` | `0, 1, ..., L_a - 1` |
| `TokenType.OPERAND_B` | `3` | Operand `b` digits `b_0, ..., b_{L_b-1}` | `0, 1, ..., L_b - 1` |
| `TokenType.CARRY` | `4` | Incoming carry `c_k` (`k`) and emitted outgoing carry `c_{k+1}` (`k + 1`) | `k` (incoming) / `k + 1` (outgoing) |
| `TokenType.OUTPUT` | `5` | Emitted sum digits `o_j` | `1..k` in [`cls`](variants/cls.md); `0..k` in all `SubspaceLMHead` variants |
| `TokenType.ADDRESS` | `6` | Base-`B` column address digits `addr(j)` in [`addressed_step_ar`](variants/addressed_step_ar.md) and [`addressed_ar`](variants/addressed_ar.md) | `j` |

| `AdditionToken` | Byte ID | Glyph | Purpose |
| :--- | :--- | :--- | :--- |
| `AdditionToken.QUERY_SUM` | `0xF8` | `[S]` | Queries or marks the sum digit `o_k` |
| `AdditionToken.QUERY_CARRY` | `0xF9` | `[C]` | Queries or marks the outgoing carry `c_{k+1}` |
| `AdditionToken.QUERY_FLAG` | `0xFA` | `[F]` | Queries the halting flag `flag_k` |
| `AdditionToken.IDX` | `0xFB` | `[IDX]` | Opens a base-`B` column-addressed entry `[IDX] addr(j) ...` |

How the step state or trace after the third `[SEP]` is laid out in each variant is specified in [`variants/cls.md`](variants/cls.md), [`variants/step_fork.md`](variants/step_fork.md), [`variants/step_ar.md`](variants/step_ar.md), [`variants/ar.md`](variants/ar.md), [`variants/addressed_step_ar.md`](variants/addressed_step_ar.md), and [`variants/addressed_ar.md`](variants/addressed_ar.md).

### Embedding modes (`EmbeddingMode`)

In `SubspaceEmbedding`, token identities (`256`-d one-hot), `token_type_ids` (`16`-d one-hot), `position_ids` (parameter-free unit-norm sinusoidal positional encodings, `d_position`-d), and a zero-initialised scratchpad (`d_scratch`-d) occupy orthogonal slices of the residual stream.

**Residual width (default: held constant)**: `d_model` is the same in every mode. `TOKEN_TYPE_PE` uses `d_model = 256 + 16 + d_position + d_scratch`; when a mode drops the token-type slice (`16`) or the position slice (`d_position`), `d_scratch` grows by the same width, so the modes differ only in what the embedding encodes, not in model capacity.

All three modes keep the exact same `input_ids` (including `[BOS]` and all three `[SEP]` delimiters) so sequence length, attention distances, and the prompt-ending `[SEP]` that triggers column `0` are held constant across ablations:

- **`TOKEN_TYPE_PE`**: attaches both `token_type_ids` and section-coupled sinusoidal `position_ids`.
- **`TOKEN_TYPE_NOPE`**: attaches `token_type_ids` only (`position_ids=None`).
- **`NOPE`**: attaches neither (`token_type_ids=None`, `position_ids=None`).

## 3. Evaluation Metrics (`experiments/addition/evaluation.py`)

Evaluation over a set of problems runs **free-running greedy rollout** across every evaluated problem (with no ground-truth carry, output, or intra-step token conditioning during evaluation) and reports seven accuracy metrics in `0.0..1.0` (`EvalMetrics`) with identical denominators across variants, together with four first-error rates (one per `FirstErrorType`), each the fraction of all evaluated problems whose first error falls in that category:

- **`exact_match`** *(per problem)*: fraction of problems where the model's unassisted rollout matches every supervised trace token, sum digit `o_k`, outgoing carry `c_{k+1}`, and halting decision across all `K + 1` columns.
- **`step_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` reference column steps across all evaluated problems where column `k` was reached during rollout with well-formed intra-column syntax (`syntax_ok`) and both the predicted sum digit and outgoing carry match `(o_k, c_{k+1})`. Columns missed after a premature halt count as incorrect against the `K + 1` reference denominator.
- **`oracle_step_accuracy`** *(per column step `k` in `0 .. K`, diagnostic in `eval_splits.json` / `eval.jsonl`)*: fraction of the `K + 1` reference column steps where both `(o_k, c_{k+1})` (and intra-column `syntax_ok`) match under single-step teacher forcing—isolating single-column transition capability when an early carry or halting error cascades through free-running `step_accuracy`.
- **`digit_accuracy`** *(per column step `k` in `0 .. K`, `K + 1` steps per problem)*: fraction of the `K + 1` reference column steps across all evaluated problems where column `k` has `syntax_ok` and the predicted sum digit matches `o_k`.
- **`carry_accuracy`** *(per carry-active column step)*: fraction of reference column steps with a non-zero incoming or outgoing carry (`c_k != s_0` or `c_{k+1} != s_0`) where column `k` has `syntax_ok` and both `(o_k, c_{k+1})` match (`None` if no evaluated column has a non-zero carry).
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model's rollout halts at the exact right column `K`—no premature halt for `k < K`, and halts immediately on completing column `K`. Scoring halting per problem rather than per step avoids class-imbalance inflation (where a model that never halts would otherwise score `K / (K + 1)`).
- **`address_accuracy`** *(per column step `k` in `0 .. K`, [`addressed_ar`](variants/addressed_ar.md) only; `None` in the other five variants)*: fraction of the `K + 1` reference column steps where segment `k` of the `addressed_ar` rollout emits the exact `[IDX] + addr(k, alphabet)` column header, isolating base-`B` address-counter generalisation (e.g. on `extrap_len`).

### First-error taxonomy (`FirstErrorType`)

For every evaluated problem that fails `exact_match`, the earliest column `k` (`0 <= k <= K`) where the model's rollout predictions deviate from the ground truth is classified into one of four mutually exclusive error categories (checking within column `k` in deterministic order: first any supervised trace structure/address/query tokens `[IDX]`, `addr(k)`, `[S]`, `[C]`, or a premature `[EOS]` before any column is completed; then the value tokens `(o_k, c_{k+1})` in emission order; and finally the halting/continuation prediction `flag_k` / `[EOS]` / next `[IDX]`):

1. **`out_of_alphabet`**: a predicted sum digit `o_k` or outgoing carry `c_{k+1}` at column `k` is not a member of the problem's `alphabet` (excluding a boundary premature `[EOS]` between columns in `ar` with `use_query_tokens=False`, which is a premature halt).
2. **`prior_intrusion`**: when evaluated on a permuted canonical alphabet (`SymbolSet.DECIMAL[:B]` or `SymbolSet.HEX[:B]`), the predicted sum digit or outgoing carry at column `k` belongs to `alphabet` and is wrong under the prompt's alphabet, but equals the symbol that conventional canonical numeral order (`'0'..'9'`, `'A'..'F'`) would assign to that digit value (`total % B` or `total // B`).
3. **`procedural_error`**: the predicted sum digit or outgoing carry belongs to `alphabet`, but has the wrong arithmetic value (and is not a `prior_intrusion`).
4. **`syntax_malformation`**: a non-value prediction at column `k` failed—either a supervised trace header/query/address token (`[IDX]`, `addr(k)`, `[S]`, `[C]`) was ill-formed, the rollout halted prematurely at a column boundary before reaching `K + 1` columns, or `(o_k, c_{k+1})` were valid and the model emitted a premature, missing, or over-running halting/step control token (`FLAG_FINISHED` / `FLAG_CONTINUE` / `[EOS]` / `[IDX]`).

Each first-error rate divides by the number of evaluated problems, not the number of failed problems, so the rates stay defined when `exact_match == 1.0` and partition the evaluated problems: `exact_match + out_of_alphabet + prior_intrusion + procedural_error + syntax_malformation == 1.0`.

**Chance baseline for `prior_intrusion`**: a wrong in-alphabet prediction is one of `B - 1` symbols, and at most one of them is the canonical symbol for that digit value, so errors with no canonical-order preference still land on it up to `1 / (B - 1)` of the time. In base `B = 2` the only wrong in-alphabet symbol is the canonical one, so every in-alphabet value error on a permuted canonical alphabet is counted as `prior_intrusion` and none as `procedural_error`; there the rate carries no evidence of a prior. Accordingly, the `permuted` evaluation split excludes `B = 2` whenever any `B >= 3` base is present in `bases` (falling back to `(2,)` only on pure-binary `bases="2"` runs), and `eval_splits.json` / `eval.jsonl` also record per-base slices (`permuted_base{b}`) so `prior_intrusion` can be compared directly against each base's `1 / (B - 1)` chance share of in-alphabet value errors (`prior_intrusion + procedural_error`).

### Evaluation schedule & artefacts (`experiments/addition/train_and_evaluate.py`)

By default (`eval_every=None`), `train_and_evaluate` evaluates all enabled splits every `max(2500, max_steps // 20)` steps (targeting 20 checkpoints across a run, floored at `2500` steps; setting `eval_every=0` runs only the final evaluation at `max_steps`), reusing each split's deterministic `seed` at every checkpoint while preserving a single continuous training stream and optimiser state:

- **`eval.jsonl`**: one JSON Lines record per evaluated checkpoint (including `max_steps`) containing `step`, `elapsed_sec`, the top-level headline pairs (`{prefix}exact_match`, `{prefix}step_accuracy`), and a nested `"splits"` mapping holding all eleven `EvalMetrics` fields per split—plus `{split}_base{b}` per-base slices whenever a split pools multiple bases (`id`, `extrap_len`, `permuted`).
- **`eval.json`**: final-step headline pairs (`{prefix}exact_match`, `{prefix}step_accuracy`) across enabled top-level splits (`exact_match` first).
- **`eval_splits.json`**: final-step full eleven-field `EvalMetrics` dictionaries keyed by split name (`id`, `ood_base3`, `ood_base12`, `extrap_len`, `permuted`) alongside `{split}_base{b}` per-base slices for multi-base splits (disaggregating per-base `prior_intrusion` baselines on `permuted` and per-base unseen-address profiles on `extrap_len`).
