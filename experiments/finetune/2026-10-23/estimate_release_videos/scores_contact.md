Contact state and timing (finger_metrics.py). Parts thumb / index / middle / ring / little / palm; contact = part total > 0.04 S (relative, ~1 N at the median S); errors in frames (16 fps). Up = finger_acc, finger_f1; down = the rest.

**all new (r10 + r8b, 19)**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 19 | 0.66 | 0.71 | 30.6 | 13.4 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 19 | 0.59 | 0.65 | 31.2 | 13.4 | 0.35 |
| E4 (estimate, mixed, 316 steps) | 19 | 0.58 | 0.63 | 39.6 | 17.1 | 0.51 |
| R8 (estimate, real only, 320 steps) | 19 | 0.53 | 0.58 | 30.2 | 13.4 | 0.49 |
| f2 (video + touch LoRA) | 19 | 0.49 | 0.62 | 41.7 | 13.5 | 0.30 |
| R6 regressor (3 seeds) | 19 | 0.94 ± 0.00 | 0.94 ± 0.01 | 7.0 ± 1.0 | 5.5 ± 0.9 | 0.12 ± 0.01 |
| frame-0 sim touch repeated | 19 | 0.80 | 0.75 | 30.4 | 13.2 | 0.30 |
| recorded replay's touch | 17 | 0.96 | 0.96 | 4.0 | 0.4 | 0.08 |

**new: r10, other scenes (7)**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 7 | 0.68 | 0.74 | 30.1 | 13.1 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 7 | 0.62 | 0.69 | 26.9 | 13.3 | 0.34 |
| E4 (estimate, mixed, 316 steps) | 7 | 0.61 | 0.67 | 35.5 | 17.5 | 0.48 |
| R8 (estimate, real only, 320 steps) | 7 | 0.57 | 0.61 | 25.3 | 13.3 | 0.48 |
| f2 (video + touch LoRA) | 7 | 0.58 | 0.67 | 35.6 | 13.3 | 0.34 |
| R6 regressor (3 seeds) | 7 | 0.87 ± 0.01 | 0.88 ± 0.01 | 13.9 ± 0.8 | 10.0 ± 1.4 | 0.18 ± 0.01 |
| frame-0 sim touch repeated | 7 | 0.78 | 0.75 | 29.7 | 13.1 | 0.27 |
| recorded replay's touch | 7 | 0.91 | 0.92 | 8.9 | 0.6 | 0.13 |

**new: r8b, box scene / top photo (12)**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 12 | 0.65 | 0.70 | 30.9 | 13.6 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 12 | 0.57 | 0.63 | 33.7 | 13.6 | 0.35 |
| E4 (estimate, mixed, 316 steps) | 12 | 0.55 | 0.61 | 42.0 | 16.9 | 0.53 |
| R8 (estimate, real only, 320 steps) | 12 | 0.52 | 0.55 | 33.0 | 13.6 | 0.50 |
| f2 (video + touch LoRA) | 12 | 0.44 | 0.60 | 45.2 | 13.6 | 0.28 |
| R6 regressor (3 seeds) | 12 | 0.97 ± 0.01 | 0.97 ± 0.01 | 3.0 ± 1.5 | 2.8 ± 2.2 | 0.08 ± 0.01 |
| frame-0 sim touch repeated | 12 | 0.81 | 0.74 | 30.8 | 13.3 | 0.32 |
| recorded replay's touch | 10 | 1.00 | 0.99 | 0.5 | 0.3 | 0.04 |

**r6 head = U8 / E / R6 training rows (6)**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 6 | 0.66 | 0.70 | 30.9 | 13.6 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 6 | 0.58 | 0.64 | 34.6 | 13.6 | 0.34 |
| E4 (estimate, mixed, 316 steps) | 6 | 0.56 | 0.62 | 41.8 | 16.6 | 0.52 |
| R8 (estimate, real only, 320 steps) | 6 | 0.52 | 0.55 | 32.6 | 13.6 | 0.48 |
| f2 (video + touch LoRA) | 6 | 0.41 | 0.58 | 47.8 | 13.6 | 0.31 |
| R6 regressor (3 seeds) | 6 | 0.99 ± 0.00 | 0.99 ± 0.00 | 0.8 ± 0.1 | 0.6 ± 0.0 | 0.07 ± 0.01 |
| frame-0 sim touch repeated | 6 | 0.81 | 0.74 | 30.8 | 13.3 | 0.32 |
| recorded replay's touch | 5 | 1.00 | 0.99 | 0.5 | 0.3 | 0.04 |

**Sensitivity: all new videos, threshold 0.02 S**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 19 | 0.65 | 0.71 | 33.2 | 13.3 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 19 | 0.59 | 0.66 | 32.7 | 13.3 | 0.35 |
| E4 (estimate, mixed, 316 steps) | 19 | 0.59 | 0.65 | 37.6 | 13.3 | 0.51 |
| R8 (estimate, real only, 320 steps) | 19 | 0.54 | 0.60 | 31.7 | 13.3 | 0.49 |
| f2 (video + touch LoRA) | 19 | 0.49 | 0.63 | 41.7 | 13.3 | 0.30 |
| R6 regressor (3 seeds) | 19 | 0.93 ± 0.01 | 0.93 ± 0.01 | 7.2 ± 1.2 | 6.2 ± 1.6 | 0.12 ± 0.01 |
| frame-0 sim touch repeated | 19 | 0.81 | 0.76 | 29.0 | 13.1 | 0.30 |
| recorded replay's touch | 17 | 0.96 | 0.97 | 3.8 | 0.3 | 0.08 |

**Sensitivity: all new videos, threshold 0.08 S**

| method | n | finger_acc | finger_f1 | press_dur_err | contact_end_err | load_share_tvd |
|---|---|---|---|---|---|---|
| U8 (both jobs, mixed, 632 steps) | 19 | 0.71 | 0.72 | 25.5 | 13.3 | 0.29 |
| E8 (estimate, mixed, 632 steps) | 19 | 0.58 | 0.62 | 28.9 | 13.6 | 0.35 |
| E4 (estimate, mixed, 316 steps) | 19 | 0.55 | 0.60 | 41.3 | 19.5 | 0.51 |
| R8 (estimate, real only, 320 steps) | 19 | 0.53 | 0.54 | 31.2 | 13.6 | 0.49 |
| f2 (video + touch LoRA) | 19 | 0.51 | 0.63 | 40.1 | 13.6 | 0.30 |
| R6 regressor (3 seeds) | 19 | 0.93 ± 0.01 | 0.93 ± 0.01 | 6.8 ± 1.1 | 3.6 ± 0.5 | 0.12 ± 0.01 |
| frame-0 sim touch repeated | 19 | 0.78 | 0.71 | 33.0 | 13.4 | 0.30 |
| recorded replay's touch | 17 | 0.96 | 0.96 | 3.9 | 0.4 | 0.08 |

