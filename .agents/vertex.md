# Vertex AI jobs

## Open: results buckets per region

Right now `vertex.toml` has a single `bucket` setting. `yessir/vertex.py` builds every job's results path from it (`/gcs/<bucket>/results/...` in the job spec, `gs://<bucket>/results/...` when submitting), whatever region the job runs in.

Louis has created three multi-region buckets, one per continent where the project has L4 training quota:

| Bucket | Location | Regions with L4 quota it covers |
| :--- | :--- | :--- |
| `yessir-results-us` | `US` | us-central1, us-east1, us-east4, us-west1 |
| `yessir-results-europe` | `EU` | europe-west2, europe-west4 |
| `yessir-results-asia` | `ASIA` | asia-south1, asia-southeast1, asia-northeast3 |

What to do, smallest first:

1. **At minimum**, switch `bucket` in `vertex.toml` to `yessir-results-us`. That covers every US region with L4 quota. `vertex.toml` is gitignored, so this is Louis's local change.
2. **Better**, pick the bucket from the job's region, so a job in Europe or Asia writes to the bucket on its own continent. The shape of the config (e.g. a region-to-bucket map in `vertex.toml`) is Louis's call. Discuss it before writing code.

Existing jobs keep working through the switch. `pull` reads each run's `gcs_run_uri` from its `vertex.json`, not from the current `bucket` setting, so runs already submitted to `north-shore-software-ca-vertex-staging` can still be pulled from there.
