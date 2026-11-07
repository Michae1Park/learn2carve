# G1 training chain: weights

Each `NN_<stage>/` folder is the last run of one G1 training stage, each warm-started from the one before.
Git keeps every run's configs, README and metrics, but only the final policy's weights
(`07_stage7_g1_7065_finish60/checkpoints/best_agent.pt`).

All the weights, stages 01–07, are also kept in a Hugging Face model repo, at the same paths as here (tag `g1-v1.0`).
`SHA256SUMS` holds each checkpoint's SHA-256; check a download from the repo root with
`cd saved/g1/chain && sha256sum -c SHA256SUMS`.

`scripts/upload_hf.py` pushes `saved/` to the Hub (a dry run unless `--upload`). It takes no token: log in once with
`hf auth login`, which keeps the token in `~/.cache/huggingface/token`, outside the repo.
