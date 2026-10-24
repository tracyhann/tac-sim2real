"""Fetch the members the dual-mode LoRA needs (head video + its timestamps, left glove, metadata, head intrinsics) for every
episode in lora_ge/episodes.json into outputs/R2S2V_mug/lora_ge/episodes/<episode_id>/. A shard already on disk
(data/TouchScale/data) is read with tarfile; any other shard is read from the Hub with HTTP range requests (HfFileSystem,
small blocks: only the 512-byte tar headers and the wanted members are downloaded, ~6.5 MB per episode instead of a
2.5 GB shard; /scratch is nearly full). The token comes from HF_HOME (never in argv). Resumable (skips complete episodes).
  HF_HOME=$R/cache/huggingface envs/armA/bin/python fetch_eps.py [--workers 6]"""
import argparse
import json
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path("/scratch/project/prj-02-phai-lab/nuo/agentic-r2s2r"); TS = R / "data/TouchScale"; OUT = R / "outputs/R2S2V_mug/lora_ge"
REPO = "datasets/2077AIDataFoundation/TouchScale"; REV = "4bc32c64c48d3711e99e1483a2520bdceda50c85"
WANT = ("rgb_head.mp4", "rgb_head.csv", "left_hand_data.npz", "json", "head_param.json")


def done(ep):
    d = OUT / "episodes" / ep
    return all((d / w).exists() for w in WANT)


def from_tar(tf, eps):
    """extract WANT members of the episodes in eps from an open tarfile (one pass over the headers)"""
    left = {e: set(WANT) for e in eps}
    for m in tf:
        e, _, suf = m.name.partition(".")
        if e in left and suf in left[e]:
            d = OUT / "episodes" / e; d.mkdir(parents=True, exist_ok=True); tmp = d / (suf + ".part")
            with tf.extractfile(m) as src, open(tmp, "wb") as dst:
                dst.write(src.read())
            tmp.rename(d / suf); left[e].discard(suf)
            if not left[e]:
                left.pop(e)
            if not left:
                break
    return left


def fetch_shard(shard, eps):
    eps = [e for e in eps if not done(e)]
    if not eps:
        return shard, "cached", {}
    local = TS / shard
    if local.exists():
        with tarfile.open(local, "r:") as tf:
            left = from_tar(tf, eps)
        return shard, "local", left
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(f"{REPO}@{REV}/{shard}", "rb", block_size=256 * 1024) as f:
        with tarfile.open(fileobj=f, mode="r:") as tf:
            left = from_tar(tf, eps)
    return shard, "hub", left


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); a = ap.parse_args()
    rows = json.loads((OUT / "episodes.json").read_text()); by = {}
    for r in rows:
        by.setdefault(r["shard"], []).append(r["episode_id"])
    with ThreadPoolExecutor(a.workers) as ex:
        for shard, how, left in ex.map(lambda kv: fetch_shard(*kv), by.items()):
            print(shard, how, "missing" if left else "ok", {k: sorted(v) for k, v in left.items()}, flush=True)
    miss = [r["episode_id"] for r in rows if not done(r["episode_id"])]
    print("episodes", len(rows), "complete", len(rows) - len(miss), "missing", miss)


if __name__ == "__main__":
    main()
