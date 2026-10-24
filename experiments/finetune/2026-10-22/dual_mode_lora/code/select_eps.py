"""Real TouchScale clips for the dual-mode (generate G / estimate E) video + touch LoRA (user 2026-10-22: one model that
generates video + paired touch from sim AND estimates touch from any given video). Picks left-hand-primary episodes of
the mug task and similar one-hand pick / place tasks (the touch panel is the LEFT glove, r2s2v/tactile_panel.py), split
by episode and by participant:
  train  actor_09 mug coaster -> desk (every left-primary episode except the two held-out twins), actor_10 plastic cup
         upright -> tray, actor_05 spray bottle out of / into the basket, actor_04 plastic cup inverted / egg out of carton
  test   actor_01 entirely (mug with water onto the coaster: a new participant on the same object type) + the held-out
         twin episodes 55494490 / 4397541c (actor_09; their twins give a sim control and a G test)
Deterministic: within a task, highest quality_score first, contact on the left glove >= 0.15 of the episode.
  envs/armA/bin/python select_eps.py   -> outputs/R2S2V_mug/lora_ge/episodes.json"""
import json
from pathlib import Path

import pandas as pd

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); TS = R / "data/TouchScale"; OUT = R / "outputs/R2S2V_mug/lora_ge"
TWINS = {"edf79415": "train", "53780879": "train", "55494490": "test", "4397541c": "test"}
# (task_name, actor, n or None = all, split, short task phrase for the prompt)
PLAN = [("Move the mug from the coaster back to the marked desk area", "actor_09", None, "train",
         "grips the white ceramic mug standing on a round coaster, lifts it, moves it to the marked area on the desk and sets it down, then lets go"),
        ("Turn the inverted, drained plastic cup upright and place it on the tray", "actor_10", 8, "train",
         "picks up an upside-down plastic cup, turns it upright and places it on a tray, then lets go"),
        ("Take the spray bottle out of the cleaning-supplies basket and place it in the work area", "actor_05", 5, "train",
         "takes a spray bottle out of a basket of cleaning supplies and stands it in the work area, then lets go"),
        ("Place the filled spray bottle upright into the cleaning-supplies basket", "actor_05", 4, "train",
         "picks up a filled spray bottle and stands it upright in a basket of cleaning supplies, then lets go"),
        ("Place the empty plastic cup inverted on the drying mat", "actor_04", 5, "train",
         "picks up an empty plastic cup, turns it over and sets it upside down on a drying mat, then lets go"),
        ("Take the egg out of the egg carton and place it in the soft-pad bowl", "actor_04", 4, "train",
         "takes an egg out of an egg carton and places it gently in a bowl with a soft pad, then lets go"),
        ("Move the mug holding a small amount of water onto the coaster", "actor_01", 8, "test",
         "grips a ceramic mug holding a little water, lifts it, moves it onto a coaster and sets it down, then lets go")]


def main():
    d = pd.read_parquet(TS / "metadata/episodes.parquet"); have = {f"data/{p.name}" for p in (TS / "data").glob("*.tar")}; rows = []
    for task, actor, n, split, phrase in PLAN:
        e = d[(d.task_name == task) & (d.actor_id == actor) & (d.primary_hand == "left") & (d.tactile_contact_frac_left >= 0.15)]
        e = e.sort_values(["quality_score", "episode_id"], ascending=[False, True])
        if actor == "actor_09":                                        # every one, the held-out twins go to test
            sel = e
        else:
            sel = e.head(n)
        for _, r in sel.iterrows():
            ep8 = r.episode_id[:8]; sp = TWINS.get(ep8, split) if actor == "actor_09" else split
            rows.append(dict(episode_id=r.episode_id, ep8=ep8, actor=actor, task=task, phrase=phrase, split=sp, twin=ep8 in TWINS,
                             shard=r.shard, local=r.shard in have, duration_s=float(r.duration_s), quality=float(r.quality_score),
                             contact_left=float(r.tactile_contact_frac_left)))
    assert len({r["episode_id"] for r in rows}) == len(rows)
    tr, te = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "test"]
    assert not {r["actor"] for r in te if not r["twin"]} & {r["actor"] for r in tr}, "a held-out participant is in train"
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "episodes.json").write_text(json.dumps(rows, indent=1))
    for sp in ("train", "test"):
        x = [r for r in rows if r["split"] == sp]
        print(sp, len(x), {a: sum(r["actor"] == a for r in x) for a in sorted({r["actor"] for r in x})}, "local", sum(r["local"] for r in x), "twins", [r["ep8"] for r in x if r["twin"]])


if __name__ == "__main__":
    main()
