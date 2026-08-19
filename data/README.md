# DMT data contract

Large data is intentionally external. Place the following files under this
directory for the main training path:

- `teacher/records.jsonl`;
- `teacher/sd_embeddings.npz`;
- `teacher/latents/*.npz`;
- `baselines/clean_rollouts.pkl` for dense reward pairing;
- `gates/naturalness.json` for the naturalness gate;
- `textures/tablecloth_gingham.png` as the clean source texture.

LIBERO assets and public policy weights remain outside the source tree. Record
source revisions and SHA256 checksums in each generated run manifest.
