"""Render an SVG ground-truth site plan, not sonar or agent predictions.

Usage from repository root:
  python -m tools.preview_scene runs/clear_sand_seed001/world.json
No third-party packages required. Existing layout.svg is regenerated.
"""
import argparse
from html import escape
import json
import math
from pathlib import Path

from sim.scene import validate_world

LABELS = {"aerial_bomb": "Aerial bomb", "sphere_mine": "Sphere mine", "rock": "Rock", "tyre": "Tyre"}
SCENARIOS = {"clear_sand": "Clear sand", "half_buried": "Partial burial", "murky": "Murky water"}
WARM, COOL = "#ffb86b", "#69cbd3"


def write_preview(world, output):
    validate_world(world)
    objects = world["objects"]
    xs, ys = [o["xyz"][0] for o in objects], [o["xyz"][1] for o in objects]
    xmin, xmax = math.floor(min(xs)/10)*10-10, math.ceil(max(xs)/10)*10+10
    ymin, ymax = math.floor(min(ys)/10)*10-10, math.ceil(max(ys)/10)*10+10
    scale = min(810/(xmax-xmin), 520/(ymax-ymin))
    pw, ph = (xmax-xmin)*scale, (ymax-ymin)*scale
    ox, oy = 70+(830-pw)/2, 230+(520-ph)/2
    def point(x, y):
        return ox+(x-xmin)*scale, oy+(ymax-y)*scale
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900" viewBox="0 0 1440 900">',
           '<title>Look Before You Dive — ground-truth scene layout</title>',
           '<desc>Development site plan of known simulated objects. Positions are to scale; object symbols are enlarged. This view is not an agent observation.</desc>',
           '<rect width="1440" height="900" fill="#0b1422"/>',
           '<g font-family="sans-serif">',
           '<text x="48" y="47" font-size="12" font-weight="700" letter-spacing="2.5" fill="#69cbd3">LOOK BEFORE YOU DIVE</text>',
           '<text x="48" y="91" font-size="34" font-weight="700" fill="#f2f5fa">Seabed survey</text>',
           f'<text x="49" y="122" font-size="16" fill="#9bafc2">{escape(SCENARIOS[world["scenario"]])} / Scene {world["seed"]:03d}</text>',
           '<rect x="1144" y="54" width="246" height="36" rx="18" fill="#22313d"/>',
           '<circle cx="1164" cy="72" r="4" fill="#ffb86b"/>',
           '<text x="1180" y="77" font-size="12" font-weight="700" letter-spacing="1.2" fill="#f7cb92">GROUND-TRUTH LAYOUT</text>',
           '<path d="M 48 150 H 1392" stroke="#233344"/>',
           '<rect x="48" y="175" width="900" height="650" rx="16" fill="#111f2e" stroke="#28394a"/>',
           '<text x="73" y="201" font-size="11" font-weight="700" letter-spacing="1.7" fill="#9bafc2">WORLD COORDINATES / METRES</text>',
           f'<rect x="{ox:.2f}" y="{oy:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="#152a37" stroke="#365162"/>']
    for x in range(int(xmin), int(xmax)+1, 10):
        px, _ = point(x, ymin)
        svg.append(f'<path d="M {px:.2f} {oy:.2f} v {ph:.2f}" stroke="#29424f" stroke-width="0.8"/>')
        svg.append(f'<text x="{px:.2f}" y="{oy+ph+22:.2f}" text-anchor="middle" font-size="12" fill="#92a9bb">{x}</text>')
    for y in range(int(ymin), int(ymax)+1, 10):
        _, py = point(xmin, y)
        svg.append(f'<path d="M {ox:.2f} {py:.2f} h {pw:.2f}" stroke="#29424f" stroke-width="0.8"/>')
        svg.append(f'<text x="{ox-13:.2f}" y="{py+4:.2f}" text-anchor="end" font-size="12" fill="#92a9bb">{y}</text>')
    for obj in objects:
        x, y = point(*obj["xyz"][:2])
        color = WARM if obj["cls"] in {"aerial_bomb", "sphere_mine"} else COOL
        angle = -math.degrees(obj["yaw"])
        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="20" fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-opacity="0.22"/>')
        svg.append(f'<g transform="translate({x:.2f},{y:.2f}) rotate({angle:.2f})">')
        if obj["cls"] == "aerial_bomb":
            svg.append(f'<rect x="-12" y="-4.5" width="24" height="9" rx="4" fill="{color}"/><path d="M -8 -7 L -3 0 L -8 7" stroke="{color}" stroke-width="2" fill="none"/>')
        elif obj["cls"] == "tyre":
            svg.append(f'<circle r="7" fill="none" stroke="{color}" stroke-width="4"/>')
        elif obj["cls"] == "sphere_mine":
            svg.append(f'<circle r="7" fill="{color}"/>')
        else:
            svg.append(f'<path d="M -9 -2 L -4 -7 L 6 -5 L 9 3 L 2 7 L -7 5 Z" fill="{color}"/>')
        # Heading glyph has a proper legend, rather than explanatory footer prose.
        svg.append('<path d="M 14 0 H 27 L 22 -3 M 27 0 L 22 3" stroke="#dceaf3" stroke-width="1.3" fill="none"/>')
        svg.append('</g>')
        svg.append(f'<rect x="{x+16:.2f}" y="{y-32:.2f}" width="36" height="22" rx="6" fill="#0b1422" stroke="#3c5365"/>')
        svg.append(f'<text x="{x+34:.2f}" y="{y-17:.2f}" text-anchor="middle" font-size="12" font-weight="700" fill="#eaf2f8">{escape(obj["id"])}</text>')
    svg.extend([f'<text x="{ox+pw/2:.2f}" y="{oy+ph+44:.2f}" text-anchor="middle" font-size="12" fill="#b6c7d5">X</text>',
                f'<text x="{ox-37:.2f}" y="{oy+ph/2:.2f}" text-anchor="middle" font-size="12" fill="#b6c7d5">Y</text>',
                '<rect x="976" y="175" width="416" height="650" rx="16" fill="#111f2e" stroke="#28394a"/>',
                '<text x="1002" y="210" font-size="11" font-weight="700" letter-spacing="1.7" fill="#9bafc2">SCENE INVENTORY</text>',
                f'<text x="1002" y="256" font-size="32" font-weight="700" fill="#f2f5fa">{len(objects):02d}</text>',
                '<text x="1060" y="253" font-size="14" fill="#a6b9c9">known objects</text>'])
    # Sample uses four objects. Overflow gets a clear indicator rather than clipping.
    for i, obj in enumerate(objects[:4]):
        y = 291+i*91
        color = WARM if obj["cls"] in {"aerial_bomb", "sphere_mine"} else COOL
        svg.append(f'<path d="M 1002 {y-11} H 1366" stroke="#26394a"/>')
        svg.append(f'<circle cx="1010" cy="{y+9}" r="4" fill="{color}"/>')
        svg.append(f'<text x="1025" y="{y+14}" font-size="15" font-weight="700" fill="#e7eff7">{escape(obj["id"])} / {escape(LABELS[obj["cls"]])}</text>')
        svg.append(f'<text x="1025" y="{y+39}" font-size="13" fill="#9eb4c5">X {obj["xyz"][0]:.1f} m   Y {obj["xyz"][1]:.1f} m</text>')
        svg.append(f'<text x="1025" y="{y+59}" font-size="12" fill="#9eb4c5">Burial {obj["burial"]:.0%}   /   Yaw {obj["yaw"]:.2f} rad</text>')
    svg.extend(['<path d="M 1002 674 H 1366" stroke="#26394a"/>',
                f'<circle cx="1010" cy="700" r="4" fill="{WARM}"/><text x="1025" y="705" font-size="12" fill="#b5c5d3">Mine-like proxy</text>',
                f'<circle cx="1190" cy="700" r="4" fill="{COOL}"/><text x="1205" y="705" font-size="12" fill="#b5c5d3">Other proxy</text>',
                '<path d="M 1004 731 H 1020 L 1015 728 M 1020 731 L 1015 734" stroke="#dceaf3" fill="none"/>',
                '<text x="1031" y="735" font-size="12" fill="#b5c5d3">Object heading</text>',
                '<text x="1002" y="777" font-size="12" fill="#819bac">Positions to scale. Symbols enlarged.</text>',
                '<text x="48" y="864" font-size="12" fill="#819bac">SIMULATED SITE / DEVELOPMENT VIEW</text>',
                '<text x="1392" y="864" text-anchor="end" font-size="12" fill="#819bac">Source: world.json</text>',
                '</g></svg>'])
    if len(objects) > 4:
        # Inventory remains honest for larger future scenes.
        svg.insert(-1, f'<text x="1002" y="800" font-family="Arial" font-size="12" fill="#b5c5d3">+ {len(objects)-4} more objects in world.json</text>')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(svg), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    world = json.loads(args.world.read_text(encoding="utf-8"))
    print(f"Wrote {write_preview(world, args.out or args.world.with_name('layout.svg'))}")
