"""Prompts for the HF25 runs (user 2026-10-24: "the text prompt describes visual features, e.g. water in the mug, fingers
pinching the mug; most videos can share one prompt").
Every row gets the same ACTION sentence (+ "Realistic video." + PANEL = BASE + PANEL); a row adds one feature sentence only when its video has a feature the touch depends on:
the mug's content (water), the grasp (claw from above, rim pinch) or the carry (shaken, very fast). The scene, the view and the
hand's look (glove / bare / sleeve) are NOT described: the first frame carries them. grip 30 % and friction x0.5 look like the
recorded carry, so they get BASE alone. PANEL = the touch-panel sentence every earlier video + touch LoRA was trained with.
Edit FEATURES / BASE here and rebuild the dataset (build_hf25.py rewrites every metadata file)."""

ACTION = "A left hand takes a ceramic mug from a round coaster, lifts it, carries it a short way, sets it down and lets go."
BASE = ACTION + " Realistic video."
PANEL = " On the right, a grey touch map of the left glove: each cell is one sensor, brighter is pressed harder; the bars below are the total force per finger."
FEATURES = {
    "water300": "The mug is filled to the brim with water.",
    "claw": "The fingertips grip the mug from above in a claw grip, curled over the rim.",
    "rimpinch": "The thumb and two fingers pinch the mug by its rim, the fingers dipped inside the mug.",
    "shake60": "The hand shakes the mug quickly from side to side while carrying it.",
    "t10": "The hand carries the mug very quickly.",
}


def prompt_for(variant):
    """variant = the sim variant of the row (recorded, grip030, fric05, water300, t10, shake60, claw, rimpinch)"""
    f = FEATURES.get(variant)
    return ACTION + (" " + f if f else "") + " Realistic video." + PANEL


if __name__ == "__main__":
    for v in ("recorded", "grip030", "fric05", "water300", "t10", "shake60", "claw", "rimpinch"):
        print(f"{v:9s} {prompt_for(v)}")
