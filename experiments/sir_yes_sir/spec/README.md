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

### Examples (`sm_mh`, the state-machine harness read through `MultiTaskHead`, turn by turn)

**Example 1 (`alphabet = ['0', '1', '2', '3']`, `x = ['1', '3', '0']`):**
```
k = 0:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] [STEP]
        out: ('2', '0', [FLAG_CONTINUE])

k = 1:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 [STEP]
        out: ('0', '1', [FLAG_CONTINUE])

k = 2:  in:  [BOS] 0 1 2 3 [SEP] 1 3 0 [SEP] 2 0 [STEP]
        out: ('1', '0', [FLAG_FINISHED])
```

**Example 2 (`alphabet = ['q', '#', '7']`, `x = ['#', '7', 'q', '7']`):**
```
k = 0:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] [STEP]
        out: ('7', 'q', [FLAG_CONTINUE])

k = 1:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 [STEP]
        out: ('q', '#', [FLAG_CONTINUE])

k = 2:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q [STEP]
        out: ('#', 'q', [FLAG_CONTINUE])

k = 3:  in:  [BOS] q # 7 [SEP] # 7 q 7 [SEP] 7 q # [STEP]
        out: ('q', '#', [FLAG_FINISHED])
```

### Alphabets (`yessir.alphabet`)

Alphabets are constructed with `create_alphabet(symbols, *, sample_size=None, rng=None)` from `SymbolSet`:
- **Random symbols**: `create_alphabet(SymbolSet.ALL, sample_size=B, rng=rng)` — `B` distinct bytes sampled uniformly without replacement from the 240-byte symbol pool (`0x00`–`0xEF`).
- **Canonical order**: `create_alphabet(SymbolSet.DECIMAL[:B])` or `create_alphabet(SymbolSet.HEX[:B])` — conventional ASCII characters in order (`'0'`–`'9'` or `'0'`–`'9'`, `'A'`–`'F'`).
- **Canonical permuted**: `create_alphabet(SymbolSet.DECIMAL[:B], rng=rng)` — the same canonical glyphs shuffled into a random order.

## 2. Input Layouts (`experiments/sir_yes_sir/inputs.py` & `experiments/sir_yes_sir/variants/`)

### 2.1 State-machine input (`variants/sm_mh/inputs.py`: `build_sm_example`)

At step `k` (with `0 <= k < L` and previous primary outputs `out_k = [o_0, ..., o_{k-1}]`), the state-machine input is a single sequence of tokens separated by the delimiter `[SEP]` and ending with the step query trigger `[STEP]`:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  o_0 ... o_{k-1}  [STEP]
```

Each position has a token in `input_ids`, and (depending on `EmbeddingMode`) optional entries in `token_type_ids` and `position_ids`. Token type ids mark which section a token belongs to (like BERT's segment ids); position ids restart at `0` in each section.

| Section | `input_ids` | `token_type_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `TokenType.SPECIAL` (`0`) | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `TokenType.ALPHABET` (`1`) | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Input `x` | `x_0, ..., x_{L-1}` | `TokenType.INPUT` (`2`) | `0, 1, ..., L - 1` |
| Delimiter | `[SEP]` | `TokenType.SPECIAL` (`0`) | `0` |
| Output history `out_k` | `o_0, ..., o_{k-1}` | `TokenType.OUTPUT` (`3`) | `0, 1, ..., k - 1` |
| Step query trigger | `[STEP]` | `TokenType.OUTPUT` (`3`) | `k` |

In `sm_mh`, `MultiTaskHead.heads[0..2]` read the hidden state at the final unpadded position (`[STEP]`) to predict `(o_k, w_k, flag_k)` in order.

### 2.2 Autoregressive input and trace (`variants/ar/inputs.py`)

In autoregressive (`ar`) mode, the problem prefix ends at the second `[SEP]` and any `trace` tokens generated so far follow in a single `TokenType.OUTPUT` section with sequential `position_ids = 0, 1, ..., len(trace) - 1`:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  trace_0 ... trace_{m-1}
```

- **Within-step token order (`StepOrder`)**: Because tokens within step `k` are emitted sequentially and condition on earlier tokens in the step, `StepOrder` selects the 3-token order per step `k` in `build_ar_trace`:
  - `StepOrder.OUT_FIRST` (`"out_first"`): `(o_k, w_k, flag_k)`
  - `StepOrder.WRAP_FIRST` (`"wrap_first"`): `(w_k, o_k, flag_k)`
- **Teacher-forced sequence labels (`build_ar_labelled_example`)**: For a prompt prefix of length `P` (`[BOS] alphabet [SEP] x [SEP]`) and a ground-truth trace of length `3L` ending with `FLAG_FINISHED`:
  - `example.input_ids` is `prefix + trace[:-1]` (length `P + 3L - 1`).
  - `labels` is pre-shifted (`(IGNORE_INDEX,) * (P - 1) + trace`, length `P + 3L - 1`), so `TiedIdentityHead` + `next_token_cross_entropy` supervise predictions from index `P - 1` (the second `[SEP]`, predicting `trace[0]`) through `trace[-2]` (predicting `trace[-1] == FLAG_FINISHED`), while masking earlier prompt positions and right-padding with `IGNORE_INDEX = -100`.
- **Rollout (`rollout_ar` and `parse_ar_trace`)**: Free-running greedy generation starts with `trace = ()` after the second `[SEP]`, appends `argmax` tokens from `TiedIdentityHead` one by one (up to `3L` tokens), stops early if a 3-token step ends with `FLAG_FINISHED`, and parses complete 3-token chunks back into canonical `(o_k, w_k, flag_k)` tuples.

### 2.3 Embedding modes (`EmbeddingMode`)

- **`TOKEN_TYPE_PE`**: attaches both `token_type_ids` and `position_ids`.
- **`TOKEN_TYPE_NOPE`**: attaches `token_type_ids` only (`position_ids=None`).
- **`NOPE`**: attaches neither (`token_type_ids=None`, `position_ids=None`).

## 3. Scoring

Evaluation over a set of problems (`evaluate_sm_mh` and `evaluate_ar`, computed under teacher forcing) reports three metrics in `0.0..1.0`:

- **`exact_match`**: fraction of problems where every step `k` in `0 .. L - 1` emits the ground-truth `(o_k, w_k, flag_k)` (`FLAG_CONTINUE` for all `k < L - 1` and `FLAG_FINISHED` at `k == L - 1`). Because teacher-forced inputs and free-running greedy rollout (`rollout_sm_mh` / `rollout_ar`) feed the exact same prefix into the model up to the first error on a problem, teacher-forced `exact_match` equals free-running greedy `exact_match`.
- **`step_accuracy`**: fraction of individual steps across all evaluated problems where all three outputs `(o_k, w_k, flag_k)` match the ground truth.
- **`halt_accuracy`**: fraction of individual steps across all evaluated problems where the halting output matches `flag_k` (`FLAG_CONTINUE` vs. `FLAG_FINISHED`).
