// Eval split names from `eval.jsonl` / `eval_splits.json`. No node imports, so
// client-side components can use it.

/**
 * Split names in the order given, without the per-base slices `addition`
 * writes for multi-base splits (`id_base2` under `id`). A name only counts
 * as a slice when its parent is also present, so `ood_base3` (no `ood` split)
 * stays a split of its own.
 */
export function topLevelSplits(names: string[]): string[] {
  const all = new Set(names);
  return names.filter((name) => {
    const match = /^(.+)_base\d+$/.exec(name);
    return !(match && all.has(match[1]));
  });
}
