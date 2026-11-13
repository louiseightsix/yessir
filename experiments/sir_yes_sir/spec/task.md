# Sir, Yes Sir! — Task Specification

This experiment is a minimal rule-following task used to test and verify the shared `yessir` machinery end to end.

It is implemented in two variants that share the task definition, input layout, and evaluation metrics below:
- [`variants/ar.md`](variants/ar.md): autoregressive next-token prediction (`SubspaceLMHead`).
- [`variants/cls.md`](variants/cls.md): parallel per-step classification (`ParallelClassificationHead`).

## 1. Task: In-Context Successor Lookup (`experiments/sir_yes_sir/reference.py`)

Each problem instance supplies an ordered alphabet in the prompt (an in-context lookup table, not few-shot demonstrations) and asks for the cyclic successor of each symbol in an input sequence:

1. **Alphabet** (`alphabet = [s_0, s_1, ..., s_{B-1}]`): an ordered list of `B >= 2` distinct byte symbols drawn from the 240-byte symbol pool (`0x00`–`0xEF`; bytes `0xF0`–`0xFF` are reserved for special tokens so all symbols and control markers fit in a single 256-token byte vocabulary). The position `i` of a symbol `s_i` in the alphabet defines its rank (`0` to `B - 1`).
2. **Input sequence** (`x = [x_0, x_1, ..., x_{L-1}]`): a sequence of `L >= 1` symbols drawn from `alphabet`.

At each position `k` in `0 .. L - 1`, where the input symbol is `x_k = s_i`, the ground-truth transition (`reference_step`) defines:

- **Successor symbol (`o_k`)**: `s_{(i + 1) mod B}` (the next symbol in `alphabet`, wrapping around from `s_{B-1}` to `s_0`).
- **Halting (`flag_k`)**: `FLAG_CONTINUE` for `k < L - 1` and `FLAG_FINISHED` at the final step `k == L - 1`. How halting is represented in the model's output depends on the variant (emitting `[EOS]` after `o_{L-1}` in [`ar`](variants/ar.md), or predicting `flag_k` on a dedicated head in [`cls`](variants/cls.md)).

A natural circuit for this task combines positional indexing (attending to `x_k` at position `k` of the input—using explicit `position_ids` in `TOKEN_TYPE_PE`, or inferring position from the causal mask in `NOPE` and `TOKEN_TYPE_NOPE`) with associative recall over the prompt's alphabet:
- **Interior symbols (`0 <= i < B - 1`)**: locating `x_k = s_i` in the alphabet and copying the token immediately after it (`s_{i+1}`) follows the standard induction-head pattern.
- **Wrap-around (`i == B - 1`)**: the token immediately after `s_{B-1}` in the prompt is `[SEP]`, not `s_0`. A naive induction head would copy `[SEP]`; the model must instead recognise the end-of-alphabet boundary and redirect lookup to `s_0` at the start of the alphabet.

### Examples

- **Example 1**: `alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']` -> `o = ['2', '0', '1']` (`'1' -> '2'`, `'3' -> '0'` wrapping around, `'0' -> '1'`).
- **Example 2**: `alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']` -> `o = ['7', 'q', '#', 'q']` (`'#' -> '7'`, `'7' -> 'q'` wrapping around, `'q' -> '#'`, `'7' -> 'q'` wrapping around).

### Alphabets (`yessir.alphabet`)

If every problem uses the same fixed alphabet (e.g. `'0', '1', '2', '3'`), a model can ignore the prompt's alphabet section altogether and memorise static token-to-token transitions (`'0' -> '1'`, `'1' -> '2'`, ...) in its weights. Randomising the alphabet order from problem to problem makes static memorisation impossible and forces the model to look up `x_k` in context.

Alphabets are constructed with `create_alphabet(symbols, *, sample_size=None, rng=None)` from `SymbolSet`:
- **Random symbols**: `create_alphabet(SymbolSet.ALL, sample_size=B, rng=rng)` — `B` distinct bytes sampled uniformly without replacement in random order from the 240-byte symbol pool (`0x00`–`0xEF`). Prevents both transition memorisation and overfitting to a small fixed set of token embeddings.
- **Canonical order**: `create_alphabet(SymbolSet.DECIMAL[:B])` or `create_alphabet(SymbolSet.HEX[:B])` — conventional ASCII characters in fixed order (`'0'`–`'9'` (`DECIMAL`) or `'0'`–`'9'`, `'A'`–`'F'` (`HEX`)), where static memorisation is possible.
- **Canonical permuted**: `create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)` — the same canonical glyphs shuffled into a random order per problem, isolating whether a model relies on in-context lookup vs. memorised glyph order.

## 2. Shared Sequence Layout & Embedding Modes (`experiments/sir_yes_sir/inputs.py`)

Both [`ar`](variants/ar.md) and [`cls`](variants/cls.md) share the same input sequence format (`build_example`). Given `alphabet`, `x`, and any output tokens `o_0, ..., o_{k-1}` emitted so far:

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

See [`variants/ar.md`](variants/ar.md) and [`variants/cls.md`](variants/cls.md) for how each variant constructs training targets and runs greedy rollout over this sequence.

### Embedding modes (`EmbeddingMode`)

In `SubspaceEmbedding`, token identities (`256`-d one-hot), `token_type_ids` (`16`-d one-hot), `position_ids` (parameter-free unit-norm sinusoidal positional encodings, `d_position`-d), and a zero-initialised scratchpad (`d_scratch`-d) occupy orthogonal slices of the residual stream.

**Residual width (default: held constant)**: `d_model` is the same in every mode. `TOKEN_TYPE_PE` uses `d_model = 256 + 16 + d_position + d_scratch`; when a mode drops the token-type slice (`16`) or the position slice (`d_position`), `d_scratch` grows by the same width, so the modes differ only in what the embedding encodes, not in model capacity.

All three modes keep the exact same `input_ids` (including `[BOS]` and both `[SEP]` delimiters) so sequence length, attention distances, and the prompt-ending `[SEP]` that triggers `o_0` at `k = 0` are held constant across ablations:

- **`TOKEN_TYPE_PE`**: attaches both `token_type_ids` and section-coupled sinusoidal `position_ids`.
- **`TOKEN_TYPE_NOPE`**: attaches `token_type_ids` only (`position_ids=None`).
- **`NOPE`**: attaches neither (`token_type_ids=None`, `position_ids=None`; Kazemnejad et al., 2023).

## 3. Evaluation Metrics (`experiments/sir_yes_sir/evaluation.py`)

Evaluation over a set of problems (computed under teacher forcing) reports three metrics in `0.0..1.0` (`EvalMetrics`) with identical denominators across variants:

- **`exact_match`** *(per problem)*: fraction of problems where every output symbol and halting prediction matches the ground truth. Because all variants use a causal attention mask and deterministic `argmax` decoding, teacher-forced inputs and free-running greedy rollout feed the exact same prefix into the model up to the first error on a problem, so teacher-forced `exact_match` equals free-running greedy `exact_match`. This equality holds in exact arithmetic; under `bfloat16` autocast, differences in padding and batch shape can flip near-tied `argmax` decisions, so the two may differ slightly.
- **`step_accuracy`** *(per input step `k` in `0 .. L - 1`, `L` steps per problem)*: fraction of the `L` steps across all evaluated problems where the predicted symbol matches `o_k`.
- **`halt_accuracy`** *(per problem)*: fraction of problems where the model halts at the exact right step—no premature halt before `o_{L-1}`, and halts immediately after `o_{L-1}`. Scoring halting per problem rather than per step avoids class-imbalance inflation (where a model that never halts would otherwise score near `1 - 1 / L`).
