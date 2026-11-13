// An experiment's spec: `experiments/<name>/spec/task.md` plus one
// `spec/variants/<variant>.md` per variant, rendered to HTML for the Spec page.
import { posix } from "node:path";

/** What Astro gives back for an imported `.md` file (only the part used here). */
interface MarkdownModule {
  compiledContent: () => string | Promise<string>;
}

// Vite imports every spec file at build time (and watches them under `astro dev`,
// so editing one reloads the page). Keys look like
// "../../../experiments/sir_yes_sir/spec/variants/ar.md".
const specModules = import.meta.glob<MarkdownModule>("../../../experiments/*/spec/**/*.md", {
  eager: true,
});

export interface SpecSection {
  /** Path inside `spec/`, e.g. "task.md" or "variants/ar.md". */
  path: string;
  /** Element id on the Spec page, e.g. "spec-variants-ar". */
  anchor: string;
  /** Rendered HTML, or null if the file doesn't exist. */
  html: string | null;
}

/** "variants/ar.md" → "spec-variants-ar". */
function anchorFor(path: string): string {
  return `spec-${path.replace(/\.md$/, "").replaceAll("/", "-")}`;
}

/**
 * Point links between spec files at their section on the page.
 *
 * task.md links to "variants/ar.md" and the variants link back to "../task.md".
 * Those relative paths mean nothing in the browser, so a link to another spec
 * file becomes "#<anchor>" for its section. A link to a heading inside it
 * ("../task.md#2-shared-...") becomes just "#2-shared-...", because Astro gives
 * every heading an id. Other links are left alone.
 */
function rewriteSpecLinks(html: string, fromPath: string, anchors: Map<string, string>): string {
  return html.replace(/href="([^"#]+\.md)(#[^"]*)?"/g, (match, href: string, fragment?: string) => {
    const target = posix.normalize(posix.join(posix.dirname(fromPath), href));
    const anchor = anchors.get(target);
    if (!anchor) return match;
    return `href="${fragment ?? `#${anchor}`}"`;
  });
}

/**
 * The spec sections for one experiment, in reading order: task.md, then each variant.
 *
 * Variants come from the experiment schema plus any `variants/*.md` on disk, so a
 * variant without a spec file still gets a section (with `html: null`).
 */
export async function loadSpec(experiment: string, variantNames: string[]): Promise<SpecSection[]> {
  const prefix = `../../../experiments/${experiment}/spec/`;
  const filesOnDisk = Object.keys(specModules)
    .filter((key) => key.startsWith(prefix))
    .map((key) => key.slice(prefix.length));

  const variantPaths = new Set(variantNames.map((v) => `variants/${v}.md`));
  for (const path of filesOnDisk) {
    if (path.startsWith("variants/")) variantPaths.add(path);
  }
  const paths = ["task.md", ...[...variantPaths].sort()];
  const anchors = new Map(paths.map((p) => [p, anchorFor(p)]));

  return Promise.all(
    paths.map(async (path) => {
      const module = specModules[prefix + path];
      const html = module ? rewriteSpecLinks(await module.compiledContent(), path, anchors) : null;
      return { path, anchor: anchorFor(path), html };
    }),
  );
}
