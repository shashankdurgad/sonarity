"""Gemini planner: reads contacts, crops and inspection images, writes targets.json."""

import base64
import json
import os
import sys

try:
    from google import genai
except ImportError:
    genai = None


SYSTEM_PROMPT = """\
You are an underwater ordnance classification assistant. You receive sonar crops
and inspection images of objects found on the seabed. For each contact, decide
whether it is UXO (unexploded ordnance), non-UXO, or uncertain.

Respond with a JSON array. Each element must have exactly these fields:
- contact_id: string matching the input
- class: one of "UXO", "non-UXO", "uncertain"
- confidence: float 0.0 to 1.0
- priority: integer, 1 = most urgent
- standoff_m: suggested exclusion radius in metres (50 for UXO, 0 for non-UXO, 25 for uncertain)
- reason: one sentence explaining your decision

Return ONLY the JSON array, no markdown fences or extra text."""


TEMPLATE_VERDICTS = {
    "mine_like": {
        "class": "UXO",
        "confidence": 0.75,
        "standoff_m": 50,
        "reason": "Sonar return consistent with mine-like object",
    },
    "other": {
        "class": "non-UXO",
        "confidence": 0.70,
        "standoff_m": 0,
        "reason": "Shape and reflectivity consistent with natural debris",
    },
}


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _load_viewpoints(run_dir: str) -> dict:
    """Load viewpoints.json and index by contact_id."""
    vp_path = os.path.join(run_dir, "viewpoints.json")
    if not os.path.exists(vp_path):
        return {}
    with open(vp_path) as f:
        return {v["contact_id"]: v for v in json.load(f)}


def _build_prompt(contacts: list, run_dir: str) -> tuple[str, list]:
    """Build the text prompt and list of image parts for Gemini."""
    viewpoints = _load_viewpoints(run_dir)
    lines = ["Classify each contact:\n"]
    images = []
    for c in contacts:
        lines.append(f"Contact {c['id']}: detected class={c['cls']}, conf={c['conf']:.2f}, "
                      f"size={c['size_m']}")
        vp = viewpoints.get(c["id"])

        # Primary image: inspection from the best angle (if available)
        if vp and "inspection" in vp:
            insp_path = os.path.join(run_dir, vp["inspection"])
            if os.path.exists(insp_path):
                images.append(("image/png", _encode_image(insp_path)))
                lines.append(f"  (inspection from {vp['chosen']} angle attached — best viewing angle)")

        # Secondary image: initial sonar crop from the survey pass
        crop_path = os.path.join(run_dir, c["crop"])
        if os.path.exists(crop_path):
            images.append(("image/png", _encode_image(crop_path)))
            lines.append(f"  (initial sonar crop attached)")

    return "\n".join(lines), images


def classify_with_gemini(contacts: list, run_dir: str, model_name: str = "gemini-3.8-flash") -> list:
    """Call Gemini to classify contacts. Returns list of target dicts."""
    if genai is None:
        raise ImportError("google-generativeai not installed")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    prompt_text, images = _build_prompt(contacts, run_dir)
    parts = [prompt_text]
    for mime, data in images:
        parts.append(genai.types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime))

    response = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    targets = json.loads(text)

    # Validate and ensure priority ordering
    for i, t in enumerate(targets):
        t.setdefault("priority", i + 1)
        assert t["class"] in ("UXO", "non-UXO", "uncertain")
        assert 0.0 <= t["confidence"] <= 1.0

    return targets


def classify_with_template(contacts: list) -> list:
    """Fallback: classify using simple rules when Gemini is unavailable."""
    targets = []
    priority = 1
    # Sort mine_like first
    sorted_contacts = sorted(contacts, key=lambda c: (0 if c["cls"] == "mine_like" else 1, -c["conf"]))
    for c in sorted_contacts:
        template = TEMPLATE_VERDICTS.get(c["cls"], TEMPLATE_VERDICTS["other"])
        targets.append({
            "contact_id": c["id"],
            "class": template["class"],
            "confidence": round(c["conf"] * template["confidence"], 2),
            "priority": priority,
            "standoff_m": template["standoff_m"],
            "reason": template["reason"],
        })
        priority += 1
    return targets


def run(run_dir: str, use_gemini: bool = True) -> list:
    """Main entry point. Reads contacts.json, writes targets.json."""
    contacts_path = os.path.join(run_dir, "contacts.json")
    with open(contacts_path) as f:
        contacts = json.load(f)

    if not contacts:
        targets = []
    elif use_gemini:
        try:
            targets = classify_with_gemini(contacts, run_dir)
        except Exception as e:
            print(f"[planner] Gemini failed ({e}), falling back to template", file=sys.stderr)
            targets = classify_with_template(contacts)
    else:
        targets = classify_with_template(contacts)

    targets_path = os.path.join(run_dir, "targets.json")
    with open(targets_path, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"[planner] Wrote {len(targets)} verdicts to {targets_path}")
    return targets


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "runs/sample"
    use_gemini = "--no-gemini" not in sys.argv
    run(run_dir, use_gemini=use_gemini)
