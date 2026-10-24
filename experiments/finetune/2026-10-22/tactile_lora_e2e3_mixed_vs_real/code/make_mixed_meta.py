"""metadata_mixed.json: the same 6 samples with the "Real recording." / "Simulated touch." tag removed (e2, mixed training)."""
import json
import sys
from pathlib import Path

ds = Path(sys.argv[1]); m = json.loads((ds / "metadata.json").read_text())
for x in m:
    for tag in ("Real recording. ", "Simulated touch. "):
        if x["text"].startswith(tag):
            x["text"] = x["text"][len(tag):]
(ds / "metadata_mixed.json").write_text(json.dumps(m, indent=1)); print(len({x["text"] for x in m}), "distinct prompts:", [x["text"][:60] for x in m][:2])
