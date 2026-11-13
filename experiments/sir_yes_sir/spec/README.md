# Sir, Yes Sir! — Specification

This experiment is a minimal rule-following task used to test and verify the shared `yessir` machinery end to end.

## 1. Task: In-Context Successor Lookup

Each problem instance supplies an ordered alphabet in the prompt (an in-context lookup table, not few-shot demonstrations) and asks for the cyclic successor of each symbol in an input sequence:

1. **Alphabet** (`alphabet = [s_0, s_1, ..., s_{B-1}]`): an ordered list of `B >= 2` distinct byte symbols drawn from the 240-byte symbol pool (`0x00`–`0xEF`; bytes `0xF0`–`0xFF` are reserved for special tokens so all symbols and control markers fit in a single 256-token byte vocabulary). The position `i` of a symbol `s_i` in the alphabet defines its rank (`0` to `B - 1`).
2. **Input sequence** (`x = [x_0, x_1, ..., x_{L-1}]`): a sequence of `L >= 1` symbols drawn from `alphabet`.

At each position `k` in `0 .. L - 1`, where the input symbol is `x_k = s_i`, the target output symbol is its cyclic successor in `alphabet`:

- **Successor symbol (`o_k`)**: `s_{(i + 1) mod B}` (the next symbol in `alphabet`, wrapping around from `s_{B-1}` to `s_0`).
- **Halting**: after `o_{L-1}`, generation halts—either by emitting `[EOS]` after `o_{L-1}` (in autoregressive `ar` mode) or via a parallel halting head emitting `FLAG_CONTINUE` for `k < L - 1` and `FLAG_FINISHED` at `k == L - 1` (in parallel classification `cls` mode).

A natural circuit for this task combines positional indexing (attending to `x_k` at position `k` of the input—using explicit `position_ids` in `TOKEN_TYPE_PE`, or inferring position from the causal mask in `NOPE` and `TOKEN_TYPE_NOPE`) with associative recall over the prompt's alphabet:
- **Interior symbols (`0 <= i < B - 1`)**: locating `x_k = s_i` in the alphabet and copying the token immediately after it (`s_{i+1}`) follows the standard induction-head pattern.
- **Wrap-around (`i == B - 1`)**: the token immediately after `s_{B-1}` in the prompt is `[SEP]`, not `s_0`. A naive induction head would copy `[SEP]`; the model must instead recognise the end-of-alphabet boundary and redirect lookup to `s_0` at the start of the alphabet.

### Examples

**Example 1 (`alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']` -> `o = ['2', '0', '1']`):**

Autoregressive (`ar`, read through `SubspaceLMHead`, one token per step + `[EOS]` at the end):
```
in:   [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP]
out:  2 0 1 [EOS]
```

Parallel classification (`cls`, read through `ParallelClassificationHead`, `(o_k, flag_k)` at each step):
```
k = 0:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP]
        out: ('2', [FLAG_CONTINUE])

k = 1:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2
        out: ('0', [FLAG_CONTINUE])

k = 2:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 0
        out: ('1', [FLAG_FINISHED])
```

**Example 2 (`alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']` -> `o = ['7', 'q', '#', 'q']`):**

Autoregressive (`ar`, read through `SubspaceLMHead`, one token per step + `[EOS]` at the end):
```
in:   [BOS] q # 7 [SEP] # 7 q 7 [SEP]
out:  7 q # q [EOS]
```

Parallel classification (`cls`, read through `ParallelClassificationHead`, `(o_k, flag_k)` at each step):
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

### Alphabets (`yessir.alphabet`)

If every problem uses the same fixed alphabet (e.g. `'0', '1', '2', '3'`), a model can ignore the prompt's alphabet section altogether and memorise static token-to-token transitions (`'0' -> '1'`, `'1' -> '2'`, ...) in its weights. Randomising the alphabet order from problem to problem makes static memorisation impossible and forces the model to look up `x_k` in context.

Alphabets are constructed with `create_alphabet(symbols, *, sample_size=None, rng=None)` from `SymbolSet`:
- **Random symbols**: `create_alphabet(SymbolSet.ALL, sample_size=B, rng=rng)` — `B` distinct bytes sampled uniformly without replacement in random order from the 240-byte symbol pool (`0x00`–`0xEF`). Prevents both transition memorisation and overfitting to a small fixed set of token embeddings.
- **Canonical order**: `create_alphabet(SymbolSet.DECIMAL[:B])` or `create_alphabet(SymbolSet.HEX[:B])` — conventional ASCII characters in fixed order (`'0'`–`'9'` (`DECIMAL`) or `'0'`–`'9'`, `'A'`–`'F'` (`HEX`)), where static memorisation is possible.
- **Canonical permuted**: `create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)` — the same canonical glyphs shuffled into a random order per problem, isolating whether a model relies on in-context lookup vs. memorised glyph order.

## 2. Input Layout & Variants (`experiments/sir_yes_sir/inputs.py` & `experiments/sir_yes_sir/variants/`)

Both `ar` and `cls` share the same input sequence format (`build_example`). Given `alphabet`, `x`, and any output tokens `o_0, ..., o_{k-1}` emitted so far:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  o_0 ... o_{k-1}
```

Each position has a token in `input_ids`, and (depending on `EmbeddingMode`) optional entries in `token_type_ids` and `position_ids`. Token type ids mark which section a token belongs to (like BERT's segment ids).

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Input `x` | `x_0, ..., x_{L-1}` | `TokenType.INPUT` (`2`) | `0, 1, ..., L - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output | `o_0, ..., o_{k-1}` | `TokenType.OUTPUT` (`3`) | `1, 2, ..., k` |

Output `position_ids` start at `1` (`o_0` at `1`, `o_{k-1}` at `k`) while the second `[SEP]` sits at `position_id = 0`. Consequently, at every step `k` in `0 .. L - 1`, the query token that predicts `o_k` (`[SEP]` at `k == 0`, `o_{k-1}` at `k >= 1`) has `position_id = k`—matching the `position_id = k` of `x_k` in `TokenType.INPUT` with zero offset (position coupling).

### 2.1 Autoregressive (`ar`, `SubspaceLMHead`)

In `ar` mode, the complete ground-truth output sequence has length `L + 1`: the `L` successor symbols `o_0, ..., o_{L-1}` followed by `[EOS]`.

- **Teacher-forced sequence labels (`build_ar_labelled_example`)**: For a prompt prefix of length `P` (`[BOS] alphabet [SEP] x [SEP]`) and ground-truth target sequence `targets = (o_0, ..., o_{L-1}, EOS)` of length `L + 1`:
  - `example.input_ids` is `prefix + (o_0, ..., o_{L-1})` (length `P + L`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + targets`, length `P + L`), so `SubspaceLMHead` + `next_token_cross_entropy` supervise predictions from index `P - 1` (the second `[SEP]` at `position_id = 0`, predicting `o_0`) through `o_{L-1}` (at `position_id = L`, predicting `EOS`), while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_ar`)**: Free-running greedy generation starts with `output = ()` after the second `[SEP]`, appends `argmax` tokens from `SubspaceLMHead` one by one (up to `L + 1` tokens), and stops as soon as `EOS` is emitted.

### 2.2 Parallel classification (`cls`, `ParallelClassificationHead`)

In `cls` mode, at each step `k` (`0 <= k < L`) with previous outputs `o_0, ..., o_{k-1}`, `ParallelClassificationHead.heads[0..1]` read the hidden state at the final unpadded position (`[SEP]` at `position_id = 0` when `k == 0`; `o_{k-1}` at `position_id = k` when `k >= 1`) to predict `(o_k, flag_k)` in parallel:
- `heads[0]` predicts the successor symbol `o_k`.
- `heads[1]` predicts the halting flag `flag_k` (`FLAG_CONTINUE` for `k < L - 1`, `FLAG_FINISHED` for `k == L - 1`).

During rollout (`rollout_cls`), `o_k` is appended to the output sequence after each step, and generation stops when `flag_k == FLAG_FINISHED` (or after `L` steps).

### 2.3 Embedding modes (`EmbeddingMode`)

In `SubspaceEmbedding`, token identities (`256`-d one-hot), `token_type_ids` (`16`-d one-hot), and `position_ids` (parameter-free unit-norm sinusoidal positional encodings) occupy orthogonal slices of the residual stream. All three modes keep the exact same `input_ids` (including `[BOS]` and both `[SEP]` delimiters) so sequence length, attention distances, and the prompt-ending `[SEP]` that triggers `o_0` at `k = 0` are held constant across ablations:

- **`TOKEN_TYPE_PE`**: attaches both `token_type_ids` and section-coupled sinusoidal `position_ids`.
- **`TOKEN_TYPE_NOPE`**: attaches `token_type_ids` only (`position_ids=None`).
- **`NOPE`**: attaches neither (`token_type_ids=None`, `position_ids=None`; Kazemnejad et al., 2023).

## 3. Evaluation Metrics

Evaluation over a set of problems (`evaluate_ar` and `evaluate_cls`, computed under teacher forcing) reports three metrics in `0.0..1.0` with identical denominators across both variants:

- **`exact_match`** *(per problem)*: fraction of problems where every prediction matches the ground truth: in `ar`, every token in `o_0, ..., o_{L-1}, EOS` matches; in `cls`, every step `k` in `0 .. L - 1` emits `(o_k, flag_k)`. Because both variants use a causal attention mask and deterministic `argmax` decoding, teacher-forced inputs and free-running greedy rollout (`rollout_ar` / `rollout_cls`) feed the exact same prefix into the model up to the first error on a problem, so teacher-forced `exact_match` equals free-running greedy `exact_match`.
- **`step_accuracy`** *(per input step `k` in `0 .. L - 1`, `L` steps per problem)*: fraction of the `L` steps across all evaluated problems where the predicted symbol matches `o_k` (`preds[k] == o_k` in `ar`; `heads[0]` matching `o_k` in `cls`).
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right step—no premature halt at `k < L - 1` (`preds[k] != EOS` in `ar`; `heads[1] == FLAG_CONTINUE` in `cls`) and halts immediately after `o_{L-1}` (`preds[L] == EOS` in `ar`; `heads[1] == FLAG_FINISHED` at `k == L - 1` in `cls`). Scoring halting per problem rather than per step avoids class-imbalance inflation (where a model that never halts would otherwise score `L / (L + 1)`).
