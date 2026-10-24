"""Report page of the dual-mode LoRA (held-out results, verdict, videos). Every number comes from the pulled data files:
data/eval/scores.json (eval_ge.py), data/eval/loss_table*.json (loss_table.py), data/dataset/scale.json (ge_scale.py).
  python3 build_report.py   -> index.html (videos expected in videos/<case>.mp4, the web copies from webify.py)"""
import html
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent; D = HERE / "data"
S = json.loads((D / "eval/scores.json").read_text()); ROWS = S["rows"]
LT = json.loads((D / "eval/loss_table.json").read_text()); LTS = json.loads((D / "eval/loss_table_sp.json").read_text())
SC = json.loads((D / "dataset/scale.json").read_text())


def num(v):
    return None if v is None or isinstance(v, bool) or (isinstance(v, float) and math.isnan(v)) else float(v)


def mean(xs):
    xs = [num(x) for x in xs]; xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def agg(sel):
    g = OrderedDict()
    for r in ROWS:
        if sel(r):
            g.setdefault(r["method"], []).append(r)
    return {m: dict(n=len({x["row"] for x in rs}), **{k: mean([x.get(k) for x in rs]) for k in KEYS}) for m, rs in g.items()}


KEYS = ["mug_iou", "hand_iou", "f1_6mm", "f1_pad", "pad_time_f1", "time_iou", "onset_ms", "total_r", "finger_r", "share_tv", "total_mae_rel", "total_mae_N_oracle"]
LOWER = {"onset_ms", "share_tv", "total_mae_rel", "total_mae_N_oracle"}
NAME = {"Up8": "U · both jobs, epoch 8", "Ep8": "E-only, epoch 8", "Ep4": "E-only, epoch 4", "Rp8": "R · E-only, real rows only, epoch 8",
        "Gp8": "G-only, epoch 8", "Gp4": "G-only, epoch 4", "regressor R6, real+sim": "R6 regressor, real + sim rows",
        "regressor R6, real only": "R6 regressor, real rows only", "twin sim touch (own scale)": "twin's own sim touch",
        "frame-0 touch repeated": "real frame-0 touch, repeated", "zero-shot A14B (video only)": "zero-shot A14B (no touch)",
        "real video (ceiling)": "real video (ceiling)", "h2 step 600 (earlier LoRA) [ABSOLUTE N scoring, panel v1]": "h2 step 600 (earlier LoRA, scored in N)"}
BASE = {"twin's own sim touch", "real frame-0 touch, repeated", "zero-shot A14B (no touch)", "real video (ceiling)", "h2 step 600 (earlier LoRA, scored in N)"}
LORA = {"Up8", "Ep8", "Ep4", "Rp8", "Gp8", "Gp4"}


def fmt(k, v):
    if v is None:
        return "–"
    return f"{v:.0f}" if k.endswith("_ms") else (f"{v:.1f}" if k.endswith("_N_oracle") else f"{v:.2f}")


def table(groups, cols, order, caption):
    """groups: [(title, {method: stats})]; bold = best model value (LoRA + regressor rows) per column within a group"""
    head = "".join(f"<th scope=col title='{html.escape(COLTIP.get(c, ''))}'>{COLNAME[c]}</th>" for c in cols)
    out = [f"<div class=tw><table><caption>{caption}</caption><thead><tr><th scope=col>method</th><th scope=col class=n>clips</th>{head}</tr></thead>"]
    for title, st in groups:
        out.append(f"<tbody><tr class=grp><th colspan={len(cols) + 2} scope=rowgroup>{title}</th></tr>")
        models = [m for m in order if m in st and (m in LORA or m.startswith("regressor"))]
        best = {}
        for c in cols:
            vals = [(st[m][c], m) for m in models if st[m][c] is not None]
            if vals:
                best[c] = (min if c in LOWER else max)(vals)[1]
        for m in order:
            if m not in st:
                continue
            nm = NAME.get(m, m); cls = " class=base" if nm in BASE else (" class=lora" if m in LORA else "")
            cells = "".join(f"<td{' class=best' if best.get(c) == m else ''}>{fmt(c, st[m][c])}</td>" for c in cols)
            out.append(f"<tr{cls}><th scope=row>{html.escape(nm)}</th><td class=n>{st[m]['n']}</td>{cells}</tr>")
        out.append("</tbody>")
    out.append("</table></div>")
    return "\n".join(out)


COLNAME = {"mug_iou": "mug IoU", "hand_iou": "hand IoU", "f1_6mm": "F1 @ 6 mm", "f1_pad": "F1 per pad", "pad_time_f1": "pad timing F1",
           "time_iou": "contact-time IoU", "onset_ms": "onset err (ms)", "total_r": "total r", "finger_r": "finger r", "share_tv": "load-share err",
           "total_mae_rel": "total MAE (×S)", "total_mae_N_oracle": "N MAE, oracle S"}
COLTIP = {"mug_iou": "generated video vs the simulation's mug silhouette (labeladh)", "f1_6mm": "contact F1, taxels pooled in 6 mm blocks per pad, relative threshold",
          "share_tv": "total variation between predicted and real finger load shares (lower is better)",
          "total_mae_rel": "MAE of the total relative force, in units of the episode scale S", "total_mae_N_oracle": "newton MAE of the total after multiplying by the true S"}

E_ORDER = ["Up8", "Ep8", "Ep4", "Rp8", "regressor R6, real+sim", "regressor R6, real only", "twin sim touch (own scale)", "frame-0 touch repeated"]
G_ORDER = ["Up8", "Gp8", "Gp4", "h2 step 600 (earlier LoRA) [ABSOLUTE N scoring, panel v1]", "zero-shot A14B (video only)", "real video (ceiling)"]
E_COLS = ["f1_6mm", "f1_pad", "time_iou", "onset_ms", "total_r", "finger_r", "share_tv", "total_mae_rel", "total_mae_N_oracle"]
G_COLS = ["mug_iou", "hand_iou", "f1_6mm", "time_iou", "total_r", "share_tv", "total_mae_rel"]

e_groups = [("New participant · actor_01 · 8 clips · Depth Anything control", agg(lambda r: r["job"] == "E" and "actor_01" in r["group"] and r["ctrl"] == "da")),
            ("Held-out twins · real video · Depth Anything control", agg(lambda r: r["job"] == "E" and "real video" in r["group"] and r["ctrl"] == "da")),
            ("Held-out twins · real video · sim control", agg(lambda r: r["job"] == "E" and "real video" in r["group"] and r["ctrl"] == "sim")),
            ("Held-out twins · zero-shot generated video · target = sim touch", agg(lambda r: r["job"] == "E" and "generated" in r["group"]))]
g_groups = [("Held-out twins · start = real frame 0 + its touch", agg(lambda r: r["job"] == "G" and r.get("touch0") is True)),
            ("Held-out twins · start = real frame 0, touch unknown", agg(lambda r: r["job"] == "G" and r.get("touch0") is False))]
E_TABLE = table(e_groups, E_COLS, E_ORDER, "Estimate (E): touch from a given video. Bold: best LoRA / regressor value in the group.")
G_TABLE = table(g_groups, G_COLS, G_ORDER, "Generate (G): video + touch from the sim control and frame 0. Bold: best LoRA value in the group.")
EG = {t: s for t, s in e_groups}; GG = {t: s for t, s in g_groups}
a01 = e_groups[0][1]; twr = e_groups[1][1]; gt = g_groups[0][1]


# ------------------------------------------------------------------ loss table (R0-R6, 3 seeds)
def lt_rows(res, variants):
    sel = {"actor_01": lambda x: x["actor"] == "actor_01" and x["ctrl"] == "da", "twins": lambda x: x["actor"] == "actor_09" and x["kind"] == "real" and x["ctrl"] == "da"}
    out = {}
    for v in variants:
        out[v] = {}
        for g, f in sel.items():
            per = [[mean([x[k] for x in res["test"][f"{v} s{s}"] if f(x)]) for k in ("total_r", "f1_6mm", "share_tv", "onset_ms", "total_mae_rel")] for s in (0, 1, 2)]
            out[v][g] = [mean([p[i] for p in per]) for i in range(5)]
    return out


LTR = lt_rows(LT, ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]); LTR.update(lt_rows(LTS, ["R0sp", "R3sp"]))
LT_NAME = {"R0": "R0 · L1", "R1": "R1 · Huber + ½ L1, active ×10", "R2": "R2 · L1 + Pearson", "R3": "R3 · L1 + time / space gradients",
           "R4": "R4 · contact BCE + gated L1", "R5": "R5 · 16 log bins, distance-weighted CE", "R6": "R6 · total L1 + OT on load share",
           "R0sp": "R0 on a softplus head", "R3sp": "R3 on a softplus head"}


def lt_table():
    h = ["<div class=tw><table><caption>Regression head, 3 seeds, held-out. Same trunk, data and steps; extra-term weight picked on an inner split (actor_10).</caption>",
         "<thead><tr><th scope=col rowspan=2>loss</th><th scope=colgroup colspan=4>actor_01 (8 clips)</th><th scope=colgroup colspan=4>held-out twins, real video</th></tr>",
         "<tr>" + "<th scope=col>total r</th><th scope=col>F1 @ 6 mm</th><th scope=col>load-share err</th><th scope=col>total MAE (×S)</th>" * 2 + "</tr></thead><tbody>"]
    for v, nm in LT_NAME.items():
        cells = ""
        for g in ("actor_01", "twins"):
            r, f, tv, on, mae = LTR[v][g]
            cells += "".join(f"<td>{'–' if x is None else (f'{x:.2f}')}</td>" for x in (r, f, tv, mae))
        cls = " class=lora" if v == "R6" else (" class=base" if v.endswith("sp") else "")
        h.append(f"<tr{cls}><th scope=row>{nm}</th>{cells}</tr>")
    h.append("</tbody></table></div>")
    return "\n".join(h)


# ------------------------------------------------------------------ scale table
def s_table():
    rows = [(k.split(": ", 1)[1], v) for k, v in SC["groups"].items() if k.startswith("actor: ")]
    label = {"actor_10": "actor_10 · cup upright (train)", "actor_04": "actor_04 · cup / egg (train)", "actor_09": "actor_09 · mug (train + held-out twins)",
             "actor_01": "actor_01 · mug with water (held out)", "actor_05": "actor_05 · spray bottle (train)", "sim": "sim replays"}
    rows.sort(key=lambda kv: kv[1]["median_S"])
    body = "".join(f"<tr><th scope=row>{label.get(a, a)}</th><td>{v['n']}</td><td>{v['median_S']:.1f}</td><td>{v['min_S']:.1f}–{v['max_S']:.1f}</td><td>{v['cv']:.2f}</td></tr>" for a, v in rows)
    return ("<div class=tw><table class=narrow><caption>Episode scale S (median total force over contact frames, whole episode)</caption><thead><tr><th scope=col>group</th>"
            "<th scope=col>episodes</th><th scope=col>median S (N)</th><th scope=col>range (N)</th><th scope=col>CV</th></tr></thead><tbody>" + body + "</tbody></table></div>")


# ------------------------------------------------------------------ videos
VID = HERE / "videos"; vids = sorted(VID.glob("*.mp4")) if VID.exists() else []
ACTOR = {r["row"]: r["group"] for r in ROWS}


def vcap(name):
    case = name[:-4]; job, rest = case.split("_", 1)
    if job == "E":
        ctrl = "sim control" if case.endswith("_sim") else "Depth Anything control"
        who = "zero-shot generated video of a held-out twin (target: sim touch)" if rest.startswith("sim_") else (
            "new participant actor_01" if not any(t in rest for t in ("55494490", "4397541c")) else "held-out twin, real video")
        clip = rest.replace("real_", "").replace("sim_", "").replace("_head_s0", "").rsplit("_", 1)[0]
        return f"<b>Estimate · {clip}</b> · {who} · {ctrl}. Columns: given video, real touch, U, E-only, R (real only), R6 regressor."
    clip = rest.replace("real_", "").rsplit("_", 1)[0]
    return f"<b>Generate · {clip}</b> · start = real frame 0 {'with' if case.endswith('_t') else 'without'} its touch. Top: sim control, real video + touch. Bottom: U, G-only."


def vid_html():
    if not vids:
        return "<p class=note>Videos are not attached to this build.</p>"
    order = sorted(vids, key=lambda p: (p.name[0] != "G", "actor" not in vcap(p.name), p.name))
    return "\n".join(f"<figure class=vid><video controls preload=none muted playsinline src='videos/{p.name}'></video><figcaption>{vcap(p.name)}</figcaption></figure>" for p in order)


# ------------------------------------------------------------------ page
def v(st, m, k):
    return fmt(k, st.get(m, {}).get(k))


PAGE = HERE / "page_template.html"
out = PAGE.read_text()
for key, val in {"E_TABLE": E_TABLE, "G_TABLE": G_TABLE, "LT_TABLE": lt_table(), "S_TABLE": s_table(), "VIDEOS": vid_html(),
                 "G_U_MUG": v(gt, "Up8", "mug_iou"), "G_G_MUG": v(gt, "Gp8", "mug_iou"), "G_G4_MUG": v(gt, "Gp4", "mug_iou"), "G_ZS_MUG": v(gt, "zero-shot A14B (video only)", "mug_iou"),
                 "G_REAL_MUG": v(gt, "real video (ceiling)", "mug_iou"),
                 "A01_U_R": v(a01, "Up8", "total_r"), "A01_E_R": v(a01, "Ep8", "total_r"), "A01_R_R": v(a01, "Rp8", "total_r"), "A01_REG_R": v(a01, "regressor R6, real+sim", "total_r"),
                 "A01_REGR_R": v(a01, "regressor R6, real only", "total_r"), "A01_E4_R": v(a01, "Ep4", "total_r"),
                 "A01_U_F1": v(a01, "Up8", "f1_6mm"), "A01_E_F1": v(a01, "Ep8", "f1_6mm"), "A01_R_F1": v(a01, "Rp8", "f1_6mm"), "A01_E4_F1": v(a01, "Ep4", "f1_6mm"),
                 "A01_REG_F1": v(a01, "regressor R6, real+sim", "f1_6mm"),
                 "A01_E_MAE": v(a01, "Ep8", "total_mae_rel"), "A01_R_MAE": v(a01, "Rp8", "total_mae_rel"), "A01_E4_MAE": v(a01, "Ep4", "total_mae_rel"),
                 "TW_REG_R": v(twr, "regressor R6, real+sim", "total_r"), "TW_REGR_R": v(twr, "regressor R6, real only", "total_r"),
                 "TW_U_R": v(twr, "Up8", "total_r"), "TW_E_R": v(twr, "Ep8", "total_r"), "TW_SIM_R": v(twr, "twin sim touch (own scale)", "total_r"),
                 "CAP_T": f"{SC['taxel_cap_rel']}", "CAP_B": f"{SC['bar_cap_rel']}", "CLIP_T": f"{100 * SC['clip_taxel']:.2f}", "CLIP_F": f"{100 * SC['clip_frames']:.2f}",
                 "S_MED": f"{SC['S_real']['median']:.1f}", "THR": f"{SC['thr_rel']}"}.items():
    out = out.replace("{{" + key + "}}", val)
assert "{{" not in out, out[out.index("{{"):out.index("{{") + 40]
(HERE / "index.html").write_text(out); print("wrote", HERE / "index.html", len(out) // 1024, "KB,", len(vids), "videos")
