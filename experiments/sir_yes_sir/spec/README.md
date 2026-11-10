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

### Alphabets

- **`symbols_random`**: `B` distinct bytes sampled uniformly without replacement from the 240-byte symbol pool (`0x00`–`0xEF`).
- **`canonical_order`**: conventional ASCII characters `'0'`–`'9'` followed by `'A'`–`'Z'`, truncated to length `B` (supports `2 <= B <= 36`).

## 2. Prompt Layout (Stage A: `sm_mh`)

At step `k` (with `0 <= k < L` and previous primary outputs `out_k = [o_0, ..., o_{k-1}]`), the state-machine prompt is a single sequence of tokens separated by the delimiter `[SEP]`:

```
[BOS]  s_0 ... s_{B-1}  [SEP]  x_0 ... x_{L-1}  [SEP]  o_0 ... o_{k-1}  [S]
```

Each position in the sequence has an entry in each of `input_ids`, `role_ids` and `position_ids`. Role ids mark which section a token belongs to, like BERT's `token_type_ids`; position ids restart at `0` in each section.

| Section | `input_ids` | `role_ids` | `position_ids` |
| :--- | :--- | :--- | :--- |
| Start marker | `[BOS]` | `ROLE_SPECIAL` | `0` |
| Alphabet | `s_0, ..., s_{B-1}` | `ROLE_ALPHABET` | `0, 1, ..., B - 1` |
| Delimiter | `[SEP]` | `ROLE_SPECIAL` | `0` |
| Input `x` | `x_0, ..., x_{L-1}` | `ROLE_INPUT` | `0, 1, ..., L - 1` |
| Delimiter | `[SEP]` | `ROLE_SPECIAL` | `0` |
| Output history `out_k` | `o_0, ..., o_{k-1}` | `ROLE_OUT` | `0, 1, ..., k - 1` |
| Step query trigger | `[S]` | `ROLE_OUT` | `k` |

In `sm_mh`, the three output heads (`head_out`, `head_wrap`, `head_flag`) read the hidden state at the final unpadded position (`[S]`) to predict `(o_k, w_k, flag_k)`.

## 3. Scoring

An instance is an **exact match** (`em`) if and only if the model:
1. emits the correct `o_k` and `w_k` at every step `k` in `0 .. L - 1`, and
2. emits `FLAG_CONTINUE` for all `k < L - 1` and `FLAG_FINISHED` at `k == L - 1`.
