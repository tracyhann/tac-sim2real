# Fine-tuning experiments (branch `VTLA`)

Layout: `<date started>/<short title>/` with `README.md`, `code/`. This branch carries the for-amir experiment (the backbone video model is referred to only as **W**)
plus the VTLA policy experiment.

| date | experiment | what | status |
|---|---|---|---|
| 2026-10-24 | [FHN](2026-10-24/FHN/README.md) ([runbook](2026-10-24/README.md)) | video + touch LoRA on W, normalised touch as values (touch head), 25 public pairs, control = scene depth + simulated hand / mug | ready to run |
| 2026-10-24 | [SVO](2026-10-24/SVO/README.md) | same, control = simulation render only (no depth) | ready to run |
| 2026-10-24 | [CDS](2026-10-24/CDS/README.md) | same, control = cube-scene depth + simulation render; shared code in [_common](2026-10-24/_common/README.md) | ready to run |
| 2026-10-25 | [VTLA](2026-10-25/vtla/README.md) | N0-VTLA policy (pi0.5 + latent tactile path) on the glove data: VTS = state-chunk actions vs VTF = force-informed actions (commanded joints + grip force); taxel encoder instead of DINOv2, the paper's Stage-1 predictor loss, 3 stages | unit + GPU smoke test passed (both arms, real weights, 25 rows) |
