"""The held-out cases of the dual-mode LoRA eval (shared by predict_ge.py, regress_ge.py, eval_ge.py; no heavy imports):
E on every held-out real clip with its Depth Anything control (+ the sim control for the twins), E on the twins' zero-shot
generated videos (target = sim touch), G on the twins from the real frame 0 with / without its touch panel."""
import json
from pathlib import Path

DS = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_ge/dataset")


def cases(only):
    rows = json.loads((DS / "metadata_test.json").read_text()); out = []
    for r in rows:
        if r["kind"] == "real":
            out.append(dict(job="E", case=f"E_{r['id']}_da", row=r["id"], control=r["control_alt_file_path"] or r["control_file_path"], ctrl="da", target="real"))
            if r.get("control_alt_file_path"):                    # twin: also its sim control (the main control)
                out.append(dict(job="E", case=f"E_{r['id']}_sim", row=r["id"], control=r["control_file_path"], ctrl="sim", target="real"))
                for t0 in (True, False):
                    out.append(dict(job="G", case=f"G_{r['id']}_{'t' if t0 else 'n'}", row=r["id"], control=r["control_file_path"], ctrl="sim", touch0=t0, target="real", clip=r["clip"]))
        else:                                                     # zero-shot generated video of a held-out twin + its sim touch
            out.append(dict(job="E", case=f"E_{r['id']}_sim", row=r["id"], control=r["control_file_path"], ctrl="sim", target="sim"))
    return [c for c in out if only is None or c["job"] == only]
