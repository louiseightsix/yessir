# Sir, Yes Sir! — Specification

This experiment is a minimal rule-following task used to test and verify the shared `yessir` machinery end to end.

## 1. Task: In-Context Successor

Each problem instance has two inputs:

1. **Alphabet** (`alphabet = [s_0, s_1, ..., s_{B-1}]`): an ordered list of `B >= 2` distinct byte symbols drawn from the symbol pool (`0x00`–`0xEF`). The position `i` of a symbol `s_i` in the alphabet defines its value (`0` to `B - 1`).
2. **Input sequence** (`x = [x_0, x_1, ..., x_{L-1}]`): a sequence of `L >= 1` symbols drawn from `alphabet`.

At each step `k` in `0 .. L - 1`, where the input symbol is `x_k = s_i`, the ground-truth transition produces three outputs:

- **Primary output (`o_k`)**: `s_{(i + 1) mod B}` (the next symbol in `alphabet`, wrapping around from `s_{B-1}` to `s_0`).
- **Wrap output (`w_k`)**: `s_1` if `i == B - 1` (wrapped around), else `s_0`.
- **Halting flag (`flag_k`)**: `FLAG_FINISHED` if `k == L - 1`, else `FLAG_CONTINUE`.

### Examples (`sm_mh`, the state-machine driver read through the multi-task head, turn by turn)

**Example 1 (`alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']`):**
```
k = 0:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] [S]
        out: ('2', '0', FLAG_CONTINUE)

k = 1:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 [S]
        out: ('0', '1', FLAG_CONTINUE)

k = 2:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 0 [S]
        out: ('1', '0', FLAG_FINISHED)
```

**Example 2 (`alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']`):**
```
k = 0:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] [S]
        out: ('7', 'q', FLAG_CONTINUE)

k = 1:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 [S]
        out: ('q', '#', FLAG_CONTINUE)

k = 2:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q [S]
        out: ('#', 'q', FLAG_CONTINUE)

k = 3:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q # [S]
        out: ('q', '#', FLAG_FINISHED)
```

### Alphabets (`yessir.alphabet`)

Alphabets are constructed with `create_alphabet(symbols, *, sample_size=None, rng=None)` from `SymbolSet`:
- **Random symbols**: `create_alphabet(SymbolSet.ALL, sample_size=B, rng=rng)` — `B` distinct bytes sampled uniformly without replacement from the 240-byte symbol pool (`0x00`–`0xEF`).
- **Canonical order**: `create_alphabet(SymbolSet.DECIMAL[:B])` or `create_alphabet(SymbolSet.HEX[:B])` — conventional ASCII characters in order (`'0'`–`'9'` or `'0'`–`'9'`, `'A'`–`'F'`).
- **Canonical permuted**: `create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)` — the same canonical glyphs shuffled into a random order.

## 2. Input Layouts (`experiments/sir_yes_sir/inputs.py`)

### 2.1 State-machine input (`build_sm_example`)

At step `k` (with `0 <= k < L` and previous primary outputs `out_k = [o_0, ..., o_{k-1}]`), the state-machine input is a single sequence of tokens separated by the delimiter `[SEP]` and ending with the step query trigger `[S]`:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  o_0 ... o_{k-1}  [S]
```

Each position has a token in `input_ids`, and (depending on `EmbeddingMode`) optional entries in `token_type_ids` and `position_ids`. Token type ids mark which section a token belongs to (like BERT's segment ids); position ids restart at `0` in each section.

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Input `x` | `x_0, ..., x_{L-1}` | `TokenType.INPUT` (`2`) | `0, 1, ..., L - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output history `out_k` | `o_0, ..., o_{k-1}` | `TokenType.OUT` (`3`) | `0, 1, ..., k - 1` |
| Step query trigger | `[S]` | `TokenType.OUT` (`3`) | `k` |

In `sm_mh`, the three output heads (`head_out`, `head_wrap`, `head_flag`) read the hidden state at the final unpadded position (`[S]`) to predict `(o_k, w_k, flag_k)`.

### 2.2 Autoregressive input (`build_ar_example`)

In autoregressive (`ar`) mode, the problem prefix ends at the second `[SEP]` and any `trace` tokens generated so far follow in a single `TokenType.OUT` section with sequential `position_ids = 0, 1, ..., len(trace) - 1`:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  trace_0 ... trace_{m-1}
```

### 2.3 Embedding modes (`EmbeddingMode`)

- **`TOKEN_TYPE_PE`**: attaches both `token_type_ids` and `position_ids`.
- **`TOKEN_TYPE_NOPE`**: attaches `token_type_ids` only (`position_ids=None`).
- **`NOPE`**: attaches neither (`token_type_ids=None`, `position_ids=None`).

## 3. Scoring

An instance is an **exact match** (`em`) if and only if the model:
1. emits the correct `o_k` and `w_k` at every step `k` in `0 .. L - 1`, and
2. emits `FLAG_CONTINUE` for all `k < L - 1` and `FLAG_FINISHED` at `k == L - 1`.
