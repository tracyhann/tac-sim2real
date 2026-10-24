"""metadata_real.json: only the real pairs (real video + real tactile) of metadata_mixed.json, untagged prompt (e3 ablation)."""
import json
import sys
from pathlib import Path

ds = Path(sys.argv[1]); m = [x for x in json.loads((ds / "metadata_mixed.json").read_text()) if x["source"] == "gt"]
(ds / "metadata_real.json").write_text(json.dumps(m, indent=1)); print(len(m), "samples:", [(x["clip"], x["source"]) for x in m])
