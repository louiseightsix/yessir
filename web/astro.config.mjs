// @ts-check
import { fileURLToPath } from "node:url";
import react from "@astrojs/react";
import { defineConfig } from "astro/config";
import astroAgentAnnotate from "astro-agent-annotate";

const resultsDir = fileURLToPath(new URL("../results/", import.meta.url));
const studiesDir = fileURLToPath(new URL("../studies/", import.meta.url));
const experimentsDir = fileURLToPath(new URL("../experiments/", import.meta.url));

/** True for a file in `results/`, `studies/` or any `experiments/<name>/spec/`. */
function isWatchedData(file) {
  if (file.startsWith(resultsDir) || file.startsWith(studiesDir)) return true;
  // experiments/<name>/spec/..., not the Python code next to it.
  return file.startsWith(experimentsDir) && file.slice(experimentsDir.length).split("/")[1] === "spec";
}

/**
 * Under `astro dev`, reload the browser when anything in `results/`, `studies/`
 * or an experiment's `spec/` changes.
 *
 * All three are outside `web/`, so the dev server doesn't watch them by default.
 * Pages re-read `results/` and `studies/` on every dev request; the spec files are
 * Vite imports, which Vite re-imports once it sees them change. Either way, a full
 * reload shows the new data. `experiments/` is watched as a whole, but only
 * changes under a `spec/` directory trigger a reload.
 * Training appends to `train.jsonl` every step, so changes are batched into at
 * most one reload per second.
 *
 * @returns {import("vite").Plugin}
 */
function reloadOnResultsChange() {
  return {
    name: "reload-on-results-change",
    configureServer(server) {
      server.watcher.add([resultsDir, studiesDir, experimentsDir]);
      /** @type {ReturnType<typeof setTimeout> | null} */
      let pending = null;
      server.watcher.on("all", (_event, file) => {
        if (!isWatchedData(file) || pending) return;
        pending = setTimeout(() => {
          pending = null;
          server.ws.send({ type: "full-reload" });
        }, 1000);
      });
    },
  };
}

// Static output: every page is rendered to HTML at build time.
export default defineConfig({
  output: "static",
  integrations: [react(), astroAgentAnnotate()],
  vite: { plugins: [reloadOnResultsChange()] },
});
