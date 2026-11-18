# Outstanding code issues

Open issues found in a code-correctness review of `experiments/addition`, `experiments/sir_yes_sir` and `yessir/`, rechecked against commit `ede2632`. All other issues from that review have been fixed in code or documented in the specs and study files.

None of these issues makes `exact_match` wrong. They affect what a comparison between variants, or a sweep, can be taken to show.

## 1. The step variants tell the model which column it is on

- **Where**: `experiments/addition/config.py`, where `AdditionStepConfig.embedding_mode` defaults to `EmbeddingMode.TOKEN_TYPE_PE`. This default is inherited by `cls`, `step_fork` and `step_ar`. (`addressed_step_ar` defaults to `nope`, and `ar` and `addressed_ar` are pure language models with no position or token-type ids.)
- **Issue**: under `token_type_pe`, every token belonging to column `k` carries `position_id = k`: the incoming carry, the step delimiter or query tokens, and `o_k`. The operand digits `a_k` and `b_k` carry the same position id. The model can find the digits for column `k` by matching position ids, without counting columns itself.
- **Why it matters**: `ar` gets no position information at all. So comparing `ar` with `step_ar` at default settings changes two things at once: whether the model or the harness carries state between columns, and whether the column index is supplied. A difference in results cannot be attributed to the first alone.
- **To decide**: this may be intended harness behaviour. If so, say so where the comparison is framed:
  - `experiments/addition/spec/variants/ar.md`
  - `experiments/addition/spec/variants/step_ar.md`
  - `studies/addition/variants-comparison.json`
  - `studies/addition/harness-vs-capacity.json`

  Alternatively, require runs in those studies to set `--embedding-mode nope`.

## 2. sir_yes_sir at `B = 2` does not test reading the alphabet order

- **Where**: `experiments/sir_yes_sir/spec/task.md` (alphabet section); `studies/sir_yes_sir/alphabet-length.json`, which sweeps the alphabet size `B` from 2.
- **Issue**: the task is cyclic successor. With two symbols, the successor of each symbol is the other one, whichever way round the alphabet is written. A model that has learnt "swap the two symbols" scores 100% without reading the alphabet section of the prompt.
- **To do**: state this in the spec and in the `alphabet-length` study, or start that sweep at `B = 3`.

## 3. The sir_yes_sir `cls` flag head has 254 unused classes

- **Where**: `experiments/sir_yes_sir/variants/cls/run.py:49`, `num_classes=(VOCAB_SIZE, VOCAB_SIZE)`.
- **Issue**: the second head predicts the halting flag, which is only ever `FLAG_CONTINUE` or `FLAG_FINISHED`, but it has 256 classes. At default settings `cls` therefore has 19,241,472 parameters against 18,978,816 for `ar`: about 263k more, almost all in the two untied linear heads.
- **Why it matters**: `studies/sir_yes_sir/minimum-model-size.json` and `studies/sir_yes_sir/cls-vs-ar.json` compare model capacity across the two variants. Neither mentions the difference.
- **To do**: either note it in those studies, or give the flag head 2 classes and map the flag tokens to class indices. The second option changes `cls` training, evaluation and tests.

## Limits of the verification

The commits since the review replaced teacher-forced evaluation with batched, KV-cached free-running rollout (`yessir/blocks.py`: `KVCache`, `build_causal_attention_mask`; `yessir/batch.py`: `collate_left_padded`). This code now produces `exact_match`, so it was checked against the simple one-problem-at-a-time `rollout_*` functions in each variant's `generation.py`.

- **Result**: on partially trained models, with exact match between 0% and 47%, there were 0 per-problem disagreements in 12,000 comparisons across all six addition variants. Batched and single-problem evaluation also gave identical exact match.
- **Coverage**:
  - Small models (3 layers), on CPU.
  - Bases 2 and 3, operand lengths up to 3.
  - `token_type_pe` and `nope` for `cls` and `step_fork`.
  - `token_type_pe` only for `step_ar`; `nope` only for `addressed_step_ar`.
  - All four `(step_order, use_query_tokens)` settings for the four language-model variants.
- **Untested**: long sequences (such as `extrap_len` in base 2, about 800 tokens), GPU runs, `token_type_nope`, and sir_yes_sir's new evaluators. The repo's own equivalence tests (`test_partially_trained_*_exact_match_matches_rollout`) compare aggregate, not per-problem, exact match on similarly small settings.
