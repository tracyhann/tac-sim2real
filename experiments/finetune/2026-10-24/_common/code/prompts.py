"""Prompts and structured parameters for the HF25 rows (v2, 2026-10-25, for the action-model data format; user: the text must
say WHAT the perturbation is, with its numbers - the first version only said "shakes quickly" and never gave 6 cm / 4 Hz).
Every row gets the same task instruction; a row adds ONE quantitative clause when its replay was perturbed or its grasp differs:
grip force, friction, water mass, speed, shake amplitude + frequency, or the grasp posture. The same numbers are also given as a
structured `params` dict (rows.json / metadata_*.json), generated from one table so text and parameters never disagree. The scene,
the camera and the hand's look are NOT described: the first frame carries them. The old touch-panel sentence is gone (the touch is
a value array, not pixels). v1 prompts (used by the FHN runs of 10-24) = BASE + a qualitative feature sentence + the panel sentence.
  python prompts.py   -> prints every variant's prompt and params"""

ACTION = "Pick up the ceramic mug from its coaster, lift it, carry it a short way, set it down and let go."
BASE = ACTION                                                     # kept for older imports
PANEL = ""                                                        # v1 appended the touch-panel sentence; v2 has no panel

# variant -> (clause, params); params are the simulation's own settings (counterfactual kinds of the cliprep harness)
_DEFAULT = dict(grip_scale=1.0, friction_scale=1.0, water_g=0, speed_scale=1.0, shake_amp_mm=0, shake_hz=0.0, posture="recorded")
VARIANTS = {
    "recorded": ("", {}),
    "grip030": ("Hold it with a light grip, about 30% of the recorded grip force.", dict(grip_scale=0.3)),
    "fric05": ("The mug is slippery: the hand-mug friction is half the normal value.", dict(friction_scale=0.5)),
    "water300": ("The mug is filled to the brim with water (300 g extra), so carry it steadily.", dict(water_g=300)),
    "t10": ("Do it fast: 10 times the recorded speed.", dict(speed_scale=10.0)),
    "shake60": ("While carrying it, shake the mug side to side, about 6 cm at 4 Hz.", dict(shake_amp_mm=60, shake_hz=4.0)),
    "claw": ("Grasp the mug from above in a claw grip, fingertips curled over the rim.", dict(posture="claw")),
    "rimpinch": ("Pinch the mug by its rim: thumb outside, two fingers dipped inside.", dict(posture="rimpinch")),
}


def params_for(variant):
    p = dict(_DEFAULT); p.update(VARIANTS[variant][1]); p["variant"] = variant
    return p


def _num(x):
    return f"{x:g}"


def clause_from_params(p):
    """the quantitative clause for ANY parameter setting (the named VARIANTS are the special cases of this): one sentence per
    changed quantity, in a fixed order, so the same numbers always read the same way"""
    out = []
    if p.get("posture", "recorded") == "claw":
        out.append("Grasp the mug from above in a claw grip, fingertips curled over the rim.")
    elif p.get("posture") == "rimpinch":
        out.append("Pinch the mug by its rim: thumb outside, two fingers dipped inside.")
    elif p.get("posture") == "pinch2":
        out.append("Pinch the mug's body between the thumb and two fingers.")
    elif p.get("posture") == "wrap":
        out.append("Wrap the whole hand around the mug's body.")
    if p.get("grip_scale", 1.0) != 1.0:
        out.append(f"Hold it with a {'light' if p['grip_scale'] < 1 else 'firm'} grip, about {round(p['grip_scale'] * 100):d}% of the recorded grip force.")
    if p.get("friction_scale", 1.0) != 1.0:
        fx = "half" if p["friction_scale"] == 0.5 else f"{_num(p['friction_scale'])} times"
        out.append(f"The mug is {'slippery' if p['friction_scale'] < 1 else 'grippy'}: the hand-mug friction is {fx} the normal value.")
    if p.get("water_g", 0):
        out.append(f"The mug {'is filled to the brim with water' if p['water_g'] >= 300 else 'holds water'} ({int(p['water_g'])} g extra), so carry it steadily.")
    if p.get("speed_scale", 1.0) != 1.0:
        out.append(f"Do it fast: {_num(p['speed_scale'])} times the recorded speed.")
    if p.get("shake_amp_mm", 0):
        axis = "up and down" if p.get("shake_axis", "lat") == "vert" else "side to side"
        out.append(f"While carrying it, shake the mug {axis}, about {_num(p['shake_amp_mm'] / 10)} cm at {_num(p['shake_hz'])} Hz.")
    return " ".join(out)


def prompt_from_params(p):
    c = clause_from_params(p)
    return ACTION + (" " + c if c else "")


def prompt_for(variant):
    """the training text of a named row: task instruction + the variant's quantitative clause (none for the recorded replay)"""
    clause = VARIANTS[variant][0]
    return ACTION + (" " + clause if clause else "")


if __name__ == "__main__":
    for v in VARIANTS:
        assert prompt_from_params(params_for(v)) == prompt_for(v), v          # the named rows are special cases of the parametric clause
        print(f"{v:9s} {prompt_for(v)}\n          {params_for(v)}")
    print(prompt_from_params(dict(_DEFAULT, shake_amp_mm=20, shake_hz=2.0, shake_axis="vert", speed_scale=1.5, water_g=100, friction_scale=0.75, grip_scale=0.5)))
