# Fine-tuning experiments (branch `for-amir`)

Layout: `<date started>/<short title>/` with `README.md`, `code/`. This branch carries one experiment, prepared for an internal
test by people outside the lab; the backbone video model is referred to only as **W**.

| date | experiment | what | status |
|---|---|---|---|
| 2026-10-24 | [FHN](2026-10-24/FHN/README.md) ([runbook](2026-10-24/README.md)) | video + touch LoRA on W, normalised touch as values (touch head), 25 public pairs, control = scene depth + simulated hand / mug | ready to run |
| 2026-10-24 | [SVO](2026-10-24/SVO/README.md) | same, control = simulation render only (no depth) | ready to run |
| 2026-10-24 | [CDS](2026-10-24/CDS/README.md) | same, control = cube-scene depth + simulation render; shared code in [_common](2026-10-24/_common/README.md) | ready to run |
