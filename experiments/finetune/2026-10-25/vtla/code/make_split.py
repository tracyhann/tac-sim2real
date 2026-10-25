"""The held-out rows of the comparison runs (README section 10), written to <dataset>/vtla/heldout.txt (one id per line):
  unseen conditions   every row of: lateral shake 40 mm at 4 Hz, grip 40 %, 300 g of water with friction x 0.5, the claw
                      posture (both source clips where they exist): the rows where command and outcome part
  unseen jitter seeds jitter rows whose seed is a multiple of 10 (near duplicates of training rows: the easy check)
Everything else trains. The eval reports the two groups separately (eval_vtla.py --ids $(grep ...)).
  python make_split.py --data <dataset>"""
import argparse
import json
from pathlib import Path

CONDITIONS = {"shake_lat_40mm_4hz", "s0.4_off", "m300_f0.5_g1", "claw"}


def heldout(rows):
    cond = [r["id"] for r in rows if r.get("variant") in CONDITIONS]
    jit = [r["id"] for r in rows if str(r.get("variant", "")).startswith("jit_s") and int(r["variant"][5:]) % 10 == 0]
    return cond, jit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); a = ap.parse_args(); ds = Path(a.data)
    rows = json.loads((ds / "metadata_vtla.json").read_text())["rows"]
    for r in rows:                                                        # v2 rows keep the variant inside params, v3 rows at the top
        r["variant"] = r.get("variant") or (r.get("params") or {}).get("variant")
        assert r["variant"], r["id"]
    cond, jit = heldout(rows)
    (ds / "vtla").mkdir(exist_ok=True); (ds / "vtla" / "heldout.txt").write_text("\n".join(cond + jit) + "\n")
    (ds / "vtla" / "heldout.json").write_text(json.dumps(dict(conditions=sorted(CONDITIONS), unseen_conditions=cond, unseen_jitter=jit), indent=1))
    print(f"{len(rows)} rows: held out {len(cond)} unseen-condition rows + {len(jit)} jitter rows; {len(rows) - len(cond) - len(jit)} train")


if __name__ == "__main__":
    main()
