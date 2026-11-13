// @ts-check
import { fileURLToPath } from "node:url";
import react from "@astrojs/react";
import { defineConfig } from "astro/config";
import astroAgentAnnotate from "astro-agent-annotate";

const resultsDir = fileURLToPath(new URL("../results/", import.meta.url));
const studiesDir = fileURLToPath(new URL("../studies/", import.meta.url));

/**
 * Under `astro dev`, reload the browser when anything in `results/` or `studies/` changes.
 *
 * Both are outside `web/`, so the dev server doesn't watch them by default.
 * Pages re-read them on every dev request, so a full reload shows the new data.
 * Training appends to `train.jsonl` every step, so changes are batched into at
 * most one reload per second.
 *
 * @returns {import("vite").Plugin}
 */
function reloadOnResultsChange() {
  return {
    name: "reload-on-results-change",
    configureServer(server) {
      server.watcher.add([resultsDir, studiesDir]);
      /** @type {ReturnType<typeof setTimeout> | null} */
      let pending = null;
      server.watcher.on("all", (_event, file) => {
        if (!(file.startsWith(resultsDir) || file.startsWith(studiesDir)) || pending) return;
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
