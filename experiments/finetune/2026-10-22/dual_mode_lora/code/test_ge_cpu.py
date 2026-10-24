"""CPU unit test of the dual-mode masks (ge_mask.py) before any GPU time:
  1. job / control draws: P(G) ~ p_gen for GE rows, P(touch0 | G) ~ p_touch0, P(alt control | E, row has one) ~ 1 - p_sim,
     G never takes the alt (Depth Anything) control of a row that has a sim control, G-only / E-only runs draw one job;
  2. pixel masks: every invariant of check_mask, and the masked input (pixel * (1 - mask), the trainer's rule) is zero
     wherever the mask is 1 - in E the whole touch panel of every frame, in G every frame after 0 (+ frame 0's panel
     without touch0);
  3. latent masks: the trainer's exact path (first frame x4, groups of 4, trilinear to the latent grid, resize_mask) gives
     a clean split at latent column 104 (= x 832 / 8) in every latent frame, so no known latent touches the panel (E) and
     latent frame 0 alone is known (G);
  4. one real GEDataset item: the control it returns is the file of the job's control (alt vs main).
  envs/videoxfun/bin/python test_ge_cpu.py [n_items]"""
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

V = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/third_party/VideoX-Fun"); sys.path.insert(0, str(V)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from videox_fun.data.dataset_image_video import ImageVideoControlDataset  # noqa: E402
import ge_mask  # noqa: E402
from ge_mask import GEDataset, check_mask, make_mask, panel_x0  # noqa: E402

DS = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r/outputs/R2S2V_mug/lora_ge/dataset")


def args(modes="GE"):
    return Namespace(ge_modes=modes, ge_p_gen=0.5, ge_p_touch0=0.5, ge_p_sim_ctrl=0.5, ge_panel_x0=832, ge_canvas_w=1088)


def dataset(modes="GE", meta="metadata_train.json"):
    d = ImageVideoControlDataset(str(DS / meta), str(DS) + "/", video_sample_size=1088, video_sample_stride=1, video_sample_n_frames=81, video_repeat=1,
                                 image_sample_size=1088, enable_bucket=True, enable_camera_info=False)
    d.__class__ = GEDataset; d.ge_setup(args(modes)); return d


def latent_known(mask, lat_shape):
    """the trainer's path from the pixel mask (b f 1 h w, 1 = unknown) to the model's mask channel (resize_mask(1 - mask))"""
    import ast
    src = (V / "scripts/wan2.2_fun/train_control_lora_ge.py").read_text(); fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "resize_mask")
    ns = dict(F=F, torch=torch); exec(compile(ast.Module([fn], []), "resize_mask", "exec"), ns); resize_mask = ns["resize_mask"]   # the trainer's own function
    mask = rearrange(mask, "b f c h w -> b c f h w").float()
    mask = torch.concat([torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), mask[:, :, 1:]], dim=2)
    mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
    cond = F.interpolate(mask[:, :1], size=lat_shape[-3:], mode="trilinear", align_corners=True)
    return resize_mask(1 - mask, torch.zeros(lat_shape)), cond


def main(n_items=2):
    rows = json.loads((DS / "metadata_train.json").read_text()); res = {}
    # 1. draws
    for modes in ("GE", "G", "E"):
        d = dataset(modes); c = Counter(); n_alt_rows = 0
        for _ in range(40):
            for r in d.dataset:
                g = d.choose(r); c[g["mode"]] += 1; c[(g["mode"], g["touch0"])] += 1; c[(g["mode"], g["ctrl"], bool(r.get("control_alt_file_path")))] += 1
                assert g["control"] == (r["control_alt_file_path"] if g["ctrl"] == "alt" else r["control_file_path"])
        tot = c["G"] + c["E"]; p = dict(P_G=c["G"] / tot, P_touch0_given_G=c[("G", True)] / max(c["G"], 1),
                                         P_alt_given_E_and_alt=c[("E", "alt", True)] / max(c[("E", "alt", True)] + c[("E", "main", True)], 1),
                                         G_alt=c[("G", "alt", True)] + c[("G", "alt", False)], E_alt_without_alt=c[("E", "alt", False)])
        res[f"draws_{modes}"] = p; print(modes, p)
        assert p["G_alt"] == 0 and p["E_alt_without_alt"] == 0
        if modes == "GE":
            assert abs(p["P_G"] - 0.5) < 0.04 and abs(p["P_touch0_given_G"] - 0.5) < 0.05 and abs(p["P_alt_given_E_and_alt"] - 0.5) < 0.06
        else:
            assert p["P_G"] == (1.0 if modes == "G" else 0.0)
    # 2. pixel masks + masked inputs
    a = args(); f, h, w = 81, 480, 1088; x0 = panel_x0(w, a); assert x0 == 832
    pv = torch.rand(f, 3, h, w) * 2 - 1
    for ge in (dict(mode="G", touch0=True), dict(mode="G", touch0=False), dict(mode="E", touch0=False)):
        m = make_mask(pv.size(), ge, a); check_mask(m, ge, x0); mp = pv * (1 - m)
        unk = m.bool().expand_as(mp); assert (mp[unk] == 0).all() and torch.equal(mp[~unk], pv[~unk])
        if ge["mode"] == "E":
            assert (mp[:, :, :, x0:] == 0).all() and torch.equal(mp[:, :, :, :x0], pv[:, :, :, :x0])
        else:
            assert (mp[1:] == 0).all() and torch.equal(mp[0, :, :, :x0], pv[0, :, :, :x0]) and ((mp[0, :, :, x0:] == 0).all() != ge["touch0"])
        # 3. latent masks (b, 16, 21, 60, 136)
        km, cond = latent_known(m[None], (1, 16, 21, h // 8, w // 8)); km = km[0, 0]; lx = x0 // 8
        if ge["mode"] == "E":
            ok = bool((km[:, :, :lx] == 1).all() and (km[:, :, lx:] == 0).all())
        else:
            ok = bool((km[0, :, :lx] == 1).all() and (km[1:] == 0).all() and ((km[0, :, lx:] == 1).all() if ge["touch0"] else (km[0, :, lx:] == 0).all()))
        res[f"latent_{ge['mode']}_{ge['touch0']}"] = dict(clean_split=ok, unique=sorted(set(km.flatten().tolist())))
        print("latent", ge, res[f"latent_{ge['mode']}_{ge['touch0']}"]); assert ok
    # 4. real items: the returned control is the chosen file
    import av
    def dec(p):
        c = av.open(str(p)); x = np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)]); c.close(); return x
    d = dataset("GE"); twin = [i for i, r in enumerate(d.dataset) if r.get("control_alt_file_path")][:n_items]
    for i in twin:
        for _ in range(6):
            s = d[i]; g = s["ge"]; ref = dec(DS / g["control"])
            diff = float(np.abs(s["control_pixel_values"].astype(np.float32) - ref.astype(np.float32)).mean())
            other = d.dataset[i]["control_alt_file_path"] if g["ctrl"] == "main" else d.dataset[i]["control_file_path"]
            dother = float(np.abs(s["control_pixel_values"].astype(np.float32) - dec(DS / other).astype(np.float32)).mean())
            print("item", d.dataset[i]["id"], g["mode"], g["ctrl"], "diff to chosen", round(diff, 3), "to the other", round(dother, 3))
            assert diff < 0.5 and dother > 2.0, (diff, dother)
            assert d.dataset[i]["control_file_path"] == rows[i]["control_file_path"], "the row was not restored"
    res["items_ok"] = True; (DS / "test_ge_cpu.json").write_text(json.dumps(res, indent=1)); print("ALL OK")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
