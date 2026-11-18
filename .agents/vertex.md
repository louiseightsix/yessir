# Vertex AI jobs

## Results buckets

`vertex.toml` has one `bucket` setting. `yessir/vertex.py` builds each job's results path from it (`/gcs/<bucket>/results/...` in the job spec, `gs://<bucket>/results/...` when submitting), whatever region the job runs in. `./yessir.sh vertex submit --bucket <name>` overrides it for one submission.

Louis has created three multi-region buckets, one per continent where the project has L4 training quota:

| Bucket | Location | Regions with L4 quota it covers |
| :--- | :--- | :--- |
| `yessir-results-us` | `US` | us-central1, us-east1, us-east4, us-west1 |
| `yessir-results-europe` | `EU` | europe-west2, europe-west4 |
| `yessir-results-asia` | `ASIA` | asia-south1, asia-southeast1, asia-northeast3 |

Since 2026-11-18, `vertex.toml` uses `yessir-results-us`. Jobs in Europe or Asia can pass `--bucket yessir-results-europe` or `--bucket yessir-results-asia`, but a job's outputs are small, so writing across continents costs little. Picking the bucket automatically from the region hasn't been needed.

Each run's `vertex.json` records its full results path (`gcs_run_uri`), and `pull` reads that rather than the current `bucket` setting. Runs submitted before the switch still pull from `north-shore-software-ca-vertex-staging`.
