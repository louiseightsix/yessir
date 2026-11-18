# Outstanding Code Issues (`HEAD` `ede2632`)

Post-fix audit of commits `be36867` and `ede2632`. Only one code correctness bug was found in the updated codebase (alongside the open `web/` UI work tracked in [addition-eval-splits-in-web.md](file:///atelier/code/workshop/yessir/.agents/addition-eval-splits-in-web.md)).

---

## 1. Correctness & Metric Bug

### `_parse_addressed_ar_rollout` shadows premature-body-`[EOS]` handling on multi-digit addresses (`k >= B`), inflating `halt_accuracy` and misclassifying `FirstErrorType`

- **Severity**: Medium (does **not** move `exact_match` or `step_accuracy`; **does** move `halt_accuracy`, `address_accuracy`, `digit_accuracy`, and `FirstErrorType` between `out_of_alphabet` and `syntax_malformation`).
- **Where**: [evaluation.py](file:///atelier/code/workshop/yessir/experiments/addition/variants/addressed_ar/evaluation.py#L178-L241) (`_parse_addressed_ar_rollout`).
- **Affected variants**: `addition` / `addressed_ar` (both `use_query_tokens=False` and `use_query_tokens=True`, whenever a rollout halts mid-column at column index `k >= B`).

#### What goes wrong

In `_parse_addressed_ar_rollout`, line 178 determines whether segment `k` contains a complete column entry using a hardcoded 1-token minimum address width:

```python
has_full_entry = len(seg_tokens) >= 1 + body_len
```

Lines 212–236 contain a dedicated `elif` branch intended to handle the case where the model emits a valid `[IDX] + addr(k)` header and then halts prematurely with `[EOS]` inside the column body (before completing all `body_len` body tokens).

When `0 <= k < B`, `len(expected_addr) == 1`, so any incomplete body has `len(seg_tokens) < 1 + body_len` and reaches the `elif` branch as intended.

However, whenever **`k >= B`** (`len(expected_addr) >= 2`, such as `k >= 2` in base 2, `k >= 4` in base 4, `k >= 8` in base 8, `k >= 10` in base 10, or any `extrap_len` column `21..40`), a segment with a **valid multi-digit `addr(k)` followed by a partial body** (or even zero body tokens when `len(expected_addr) >= 3`, e.g. `k >= 4` in base 2 with `use_query_tokens=False`) can satisfy `len(seg_tokens) >= 1 + body_len` while still having `len(seg_tokens) < len(expected_addr) + body_len`:

1. `has_full_entry` evaluates to **`True`**, shadowing the premature-body-`[EOS]` `elif` branch at lines 212–236.
2. Lines 197–198 slice `pred_addr = tuple(seg_tokens[:-body_len])` and `body = seg_tokens[-body_len:]`, stealing the trailing address token(s) into `body` and leaving a truncated `pred_addr != expected_addr`.
3. Consequently:
   - `address_ok` is set to **`False`** instead of **`True`** (undercounting `address_accuracy`, and zeroing `digit_accuracy` on that step via `syntax_ok = False`).
   - `classify_first_error` classifies the problem as **`syntax_malformation`** instead of **`out_of_alphabet`** (unlike the identical premature `[EOS]` at `k < B`).
   - **Worst of all, when `k == last_k` (`k == K`, the final reference column)**, lines 186–192 compute:
     ```python
     halt_ok = (
         starts_with_idx
         and has_full_entry
         and (num_segments == num_steps)
         and ended_with_eos
     )
     ```
     which evaluates to **`True`**—reporting **`halt_accuracy = 1.0`** even though the model halted mid-column before emitting `c_{K+1}` (or even before emitting `o_K` when `len(expected_addr) >= 3`)!

#### Concrete triggers (verified in Python)

Let `alphabet = b"01"` (`B = 2`):

1. **Missing final carry at `k = K` (`use_query_tokens=False`)**: On `b"111" + b"1"` (`7 + 1 = 8`, `4` steps `k = 0..3`, `addr(3) = b"11"`), if columns `0..2` are right and at `k = 3` the model emits `[IDX, '1', '1', o_3, EOS]` (valid 2-bit address and valid `o_3`, halting 1 token early before `c_4`):
   - **Actual**: `halt_accuracy = 1.0`, `address_accuracy = 0.75`, `digit_accuracy = 0.75`, `syntax_malformation = 1.0`, `out_of_alphabet = 0.0` (`exact_match = 0.0`, `step_accuracy = 0.75`).
   - **Expected (matching `k = 0` on a 1-bit address)**: `halt_accuracy = 0.0`, `address_accuracy = 1.0`, `digit_accuracy = 1.0`, `syntax_malformation = 0.0`, `out_of_alphabet = 1.0` (`exact_match = 0.0`, `step_accuracy = 0.75`).
2. **Missing final carry at `k = K` (`use_query_tokens=True`)**: Same problem with `[IDX, '1', '1', [S], o_3, [C], EOS]` (`len(seg_tokens) = 5 >= 1 + 4 = 5`):
   - **Actual**: `halt_accuracy = 1.0`, `address_accuracy = 0.75`, `syntax_malformation = 1.0`, `out_of_alphabet = 0.0`.
   - **Expected**: `halt_accuracy = 0.0`, `address_accuracy = 1.0`, `syntax_malformation = 0.0`, `out_of_alphabet = 1.0`.
3. **Header-only halt on a 3-digit address (`k = 4 == K` in base 2, `use_query_tokens=False`)**: On `b"1111" + b"1"` (`5` steps `k = 0..4`, `addr(4) = b"001"`, length `3`), emitting `[IDX, '0', '0', '1', EOS]` with **zero** body tokens (`len(seg_tokens) = 3 >= 1 + 2 = 3`) still reports **`halt_accuracy = 1.0`** and `address_accuracy = 0.80` instead of `halt_accuracy = 0.0` and `address_accuracy = 1.0`.

#### Suggested fix (preserving wrong-width address handling on non-final segments)

Note why a naive `has_full_entry = len(seg_tokens) >= len(expected_addr) + body_len` across *all* segments would break a different case: on a **non-final** segment (`not is_last_seg`, where the model emitted a wrong 1-digit address `[IDX, '0', o_2, c_3]` and then continued with `[IDX, ...]` to column `3`), `seg_tokens` genuinely is a completed wrong-address column entry (`address_ok = False`, `syntax_ok = False`, `halt_ok = True`).

Instead, detect the premature-body-`[EOS]` prefix match first (before `has_full_entry`), so `has_full_entry` is `False` whenever the final `[EOS]`-terminated segment starts with `expected_addr` and ends before `len(expected_addr) + body_len`:

```python
premature_body_eos = (
    is_last_seg
    and ended_with_eos
    and starts_with_idx
    and expected_addr is not None
    and len(expected_addr) <= len(seg_tokens) < len(expected_addr) + body_len
    and tuple(seg_tokens[: len(expected_addr)]) == expected_addr
)
has_full_entry = (len(seg_tokens) >= 1 + body_len) and not premature_body_eos
```

*(Also consider whether a final `[EOS]`-terminated segment at `k == last_k` whose wrong-width address makes `len(seg_tokens) != len(expected_addr) + body_len` should still count as `halt_ok = True` or require `address_ok` / exact length; with `premature_body_eos` handled first, every valid-header premature halt across all address widths `>= 1` gets `halt_ok = False`, `address_ok = True`, and consistent `FirstErrorType` classification.)*
