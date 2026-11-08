"""Cinematic RTX replay of a finished mission. Runs inside Isaac's Python:

    /isaac-sim/python.sh render/beauty.py --run /runs/<id> --out /runs/<id>/beauty --start 0 --end 200

Rebuilds the mission's world with show materials (textured sand + animated caustics, painted and
rusted ordnance, a detailed AUV with headlights), adds life (fish schools, marine snow, kelp,
rocks, prop bubbles), underwater volumetric fog, and in-world mission graphics (sonar fan,
keep-out domes, route trail, contact beacons coloured by verdict). The robot and every mission
event follow mission_log.jsonl exactly (see render/timeline.py): nothing is re-decided here.
"""
import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=-1)
ap.add_argument("--frames", default="", help="comma list of frames (look test)")
ap.add_argument("--width", type=int, default=1920)
ap.add_argument("--height", type=int, default=1080)
ap.add_argument("--mode", default="PathTracing", help="PathTracing | RaytracedLighting | RealTimePathTracing")
ap.add_argument("--spp", type=int, default=48, help="subframes per frame (accumulation)")
ap.add_argument("--variants", default="", help="look test: comma list of render modes to compare")
ap.add_argument("--dump-settings", action="store_true")
args, _ = ap.parse_known_args()

import numpy as np  # noqa: E402

from render.timeline import FPS, Timeline  # noqa: E402

TL = Timeline(args.run)
W, H = args.width, args.height
os.makedirs(args.out, exist_ok=True)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": W, "height": H, "anti_aliasing": 0})
import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
import omni.replicator.core as rep  # noqa: E402

stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
settings = carb.settings.get_settings()
TEX = os.path.join(args.out, "_tex")
os.makedirs(TEX, exist_ok=True)
RNG = np.random.default_rng(7)
t_start = time.time()


def log(*a):
    print(f"[beauty {time.time() - t_start:6.1f}s]", *a, flush=True)


# ============================================================ render settings
def set_s(k, v):
    try:
        settings.set(k, v)
    except Exception as e:  # noqa: BLE001
        log("setting failed", k, e)


WATER = (0.02, 0.20, 0.30)


def apply_render_settings(mode: str, murk: float = 0.0):
    set_s("/rtx/rendermode", mode)
    set_s("/rtx/pathtracing/spp", 1)
    set_s("/rtx/pathtracing/totalSpp", args.spp)
    set_s("/rtx/pathtracing/maxBounces", 6)
    set_s("/rtx/pathtracing/maxSpecularAndTransmissionBounces", 6)
    set_s("/rtx/pathtracing/optixDenoiser/enabled", True)
    set_s("/rtx/pathtracing/fireflyFilter/enabled", True)
    set_s("/rtx/raytracing/fractionalCutoutOpacity", True)
    set_s("/rtx/translucency/enabled", True)
    set_s("/rtx/reflections/enabled", True)
    set_s("/rtx/indirectDiffuse/enabled", True)
    set_s("/rtx/ambientOcclusion/enabled", True)
    set_s("/rtx/shadows/enabled", True)
    set_s("/rtx/directLighting/sampledLighting/enabled", True)
    set_s("/rtx/post/histogram/enabled", True)        # auto exposure
    set_s("/rtx/post/histogram/whiteScale", 9.0)
    set_s("/rtx/post/tonemap/op", 6)                  # ACES-like filmic
    set_s("/rtx/post/lensFlares/enabled", False)
    set_s("/rtx/post/motionblur/enabled", False)
    set_s("/rtx/post/dlss/execMode", 2)
    # underwater: distance fog (rasterised modes) ...
    fog_end = 24.0 - 14.0 * murk
    set_s("/rtx/fog/enabled", True)
    set_s("/rtx/fog/fogColor", list(WATER))
    set_s("/rtx/fog/fogColorIntensity", 1.0)
    set_s("/rtx/fog/fogStartDist", 1.0)
    set_s("/rtx/fog/fogEndDist", fog_end)
    set_s("/rtx/fog/fogDistanceDensity", 1.0)
    # ... volumetric in-scattering (light shafts from the sun and the AUV headlights) ...
    set_s("/rtx/raytracing/inscattering/enabled", True)
    set_s("/rtx/raytracing/inscattering/densityMult", 1.6 + 2.5 * murk)
    set_s("/rtx/raytracing/inscattering/transmittanceColor", [0.30, 0.78, 0.88])
    set_s("/rtx/raytracing/inscattering/transmittanceMeasurementDistance", 26.0 - 14.0 * murk)
    set_s("/rtx/raytracing/inscattering/singleScatteringAlbedo", [0.55, 0.85, 0.9])
    set_s("/rtx/raytracing/inscattering/anisotropyFactor", 0.55)
    set_s("/rtx/raytracing/inscattering/maxAccumulationFrames", 32)
    set_s("/rtx/raytracing/inscattering/useDetailNoise", True)
    # ... and the path tracer's participating medium.
    set_s("/rtx/pathtracing/ptvol/enabled", True)
    set_s("/rtx/pathtracing/ptvol/transmittanceColor", [0.30, 0.78, 0.88])
    set_s("/rtx/pathtracing/ptvol/transmittanceDistance", 26.0 - 14.0 * murk)
    set_s("/rtx/pathtracing/ptvol/scatteringColor", [0.35, 0.80, 0.90])
    set_s("/rtx/pathtracing/ptvol/density", 0.10 + 0.25 * murk)
    set_s("/rtx/pathtracing/ptvol/anisotropy", 0.55)
    set_s("/rtx/pathtracing/ptvol/maxCollisionCount", 32)


apply_render_settings(args.mode)

if args.dump_settings:
    import re as _re

    def walk(prefix, d, out):
        if isinstance(d, dict):
            for k, v in d.items():
                walk(prefix + "/" + k, v, out)
        else:
            out.append((prefix, d))

    flat = []
    try:
        walk("/rtx", settings.get_settings_dictionary("/rtx").get_dict(), flat)
    except Exception:
        try:
            walk("/rtx", settings.get("/rtx"), flat)
        except Exception as e:  # noqa: BLE001
            log("dump failed", e)
    for k, v in flat:
        if _re.search(r"fog|inscatter|ptvol|volum|rendermode|histogram|tonemap", k):
            print("SETTING", k, "=", str(v)[:80])


# ============================================================ textures (procedural)
def _tile_noise(n, freqs, rng, amp_pow=1.2):
    u, v = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="xy")
    out = np.zeros((n, n))
    for _ in range(freqs):
        kx, ky = rng.integers(-24, 25, 2)
        if kx == 0 and ky == 0:
            continue
        k = math.hypot(kx, ky)
        out += np.cos(2 * math.pi * (kx * u + ky * v) + rng.uniform(0, 2 * math.pi)) / k ** amp_pow
    out -= out.min()
    return out / (out.max() + 1e-9)


def make_textures():
    n = 1024
    u, v = np.meshgrid(np.arange(n) / n, np.arange(n) / n, indexing="xy")
    noise = _tile_noise(n, 140, RNG)
    fine = _tile_noise(n, 300, np.random.default_rng(3), amp_pow=0.6)
    ripple_phase = 2 * math.pi * (9 * u + 1 * v) + 1.6 * np.sin(2 * math.pi * (2 * v + 1 * u)) + 2.5 * noise
    ripples = 0.5 + 0.5 * np.sin(ripple_phase)
    ripples = ripples ** 1.6
    height = 0.65 * ripples + 0.25 * noise + 0.10 * fine
    # albedo: warm sand, darker troughs, shell/grit specks
    base = np.array([0.74, 0.66, 0.50])
    shade = 0.72 + 0.22 * height + 0.10 * (fine - 0.5)
    alb = base[None, None, :] * shade[..., None]
    specks = (np.random.default_rng(5).random((n, n)) > 0.9965)
    alb[specks] = [0.92, 0.88, 0.80]
    dark = (np.random.default_rng(6).random((n, n)) > 0.997)
    alb[dark] = [0.25, 0.22, 0.18]
    Image.fromarray((np.clip(alb, 0, 1) * 255).astype(np.uint8)).save(f"{TEX}/sand_albedo.png")
    # normal map from height
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 3.5
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 3.5
    nrm = np.stack([-gx, gy, np.ones_like(gx)], -1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    Image.fromarray(((nrm * 0.5 + 0.5) * 255).astype(np.uint8)).save(f"{TEX}/sand_normal.png")
    # rust texture for ordnance
    rn = _tile_noise(512, 120, np.random.default_rng(11), amp_pow=0.9)
    rf = _tile_noise(512, 260, np.random.default_rng(12), amp_pow=0.5)
    paint = np.array([0.34, 0.38, 0.27])
    rust = np.array([0.58, 0.30, 0.12])
    m = np.clip((rn * 0.7 + rf * 0.3 - 0.45) * 3.0, 0, 1)[..., None]
    col = paint * (1 - m) + rust * m
    col *= (0.85 + 0.25 * rf[..., None])
    Image.fromarray((np.clip(col, 0, 1) * 255).astype(np.uint8)).save(f"{TEX}/rust.png")
    # animated caustics: 24 frames, tileable
    n2 = 512
    u2, v2 = np.meshgrid(np.arange(n2) / n2, np.arange(n2) / n2, indexing="xy")
    dirs = [(2, 1), (-1, 2), (1, -2)]
    for i in range(24):
        t = i / 24.0
        acc = np.ones((n2, n2))
        for j, (a, b) in enumerate(dirs):
            warp = 0.45 * np.sin(2 * math.pi * ((b * u2 - a * v2) + t + j / 3))
            w = np.abs(np.sin(math.pi * ((a * u2 + b * v2) + warp + t * (1 + j % 2))))
            acc = np.minimum(acc, w)
        c = np.clip(1.0 - acc, 0, 1) ** 6
        rgb = np.stack([c * 0.55, c * 0.85, c * 0.95], -1) * 0.30
        im = Image.fromarray((rgb * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2))
        im.save(f"{TEX}/caustic_{i:02d}.png")


make_textures()
log("textures ready")


# ============================================================ materials
LOOKS = "/World/Looks"
UsdGeom.Scope.Define(stage, LOOKS)
_mats = {}


def material(name, color=(0.5, 0.5, 0.5), rough=0.6, metal=0.0, emissive=None, opacity=1.0,
             albedo_tex=None, normal_tex=None, emissive_tex=None, clearcoat=0.0, ior=None):
    if name in _mats:
        return _mats[name]
    path = f"{LOOKS}/{name}"
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    st = None

    def tex(fname, out="rgb", raw=False, tag="t"):
        nonlocal st
        if st is None:
            st = UsdShade.Shader.Define(stage, path + "/st")
            st.CreateIdAttr("UsdPrimvarReader_float2")
            st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        t = UsdShade.Shader.Define(stage, f"{path}/{tag}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(fname)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if raw:
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
            t.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
        t.CreateOutput(out, Sdf.ValueTypeNames.Float3)
        return t

    if albedo_tex:
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex(albedo_tex, tag="alb").ConnectableAPI(), "rgb")
    else:
        s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    if normal_tex:
        s.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(tex(normal_tex, raw=True, tag="nrm").ConnectableAPI(), "rgb")
    if emissive_tex:
        et = tex(emissive_tex, tag="emi")
        s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(et.ConnectableAPI(), "rgb")
        _mats[name + ":emi"] = et
    elif emissive is not None:
        s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if clearcoat:
        s.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(clearcoat)
        s.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(0.05)
    if opacity < 1.0:
        s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        s.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
    if ior:
        s.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    _mats[name] = m
    _mats[name + ":shader"] = s
    return m


def bind(prim, m):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(m, UsdShade.Tokens.strongerThanDescendants)


def xform(path, t=(0, 0, 0), rz=0.0, s=(1, 1, 1)):
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*t))
    x.AddRotateZOp().Set(float(rz))
    x.AddScaleOp().Set(Gf.Vec3f(*s))
    return x


def set_xf(prim_or_x, t=None, rz=None, s=None, rxyz=None):
    x = prim_or_x if isinstance(prim_or_x, UsdGeom.Xformable) else UsdGeom.Xformable(prim_or_x)
    ops = {op.GetOpName(): op for op in x.GetOrderedXformOps()}
    if t is not None:
        ops["xformOp:translate"].Set(Gf.Vec3d(*map(float, t)))
    if rz is not None:
        ops["xformOp:rotateZ"].Set(float(rz))
    if s is not None:
        ops["xformOp:scale"].Set(Gf.Vec3f(*map(float, s)))


def mesh(path, pts, faces, counts=None, st=None, normals=None, double=False):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in np.ravel(faces)]))
    m.CreateFaceVertexCountsAttr(Vt.IntArray(counts if counts is not None else [len(faces[0])] * len(faces)))
    m.CreateSubdivisionSchemeAttr("none")
    if st is not None:
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray([Gf.Vec2f(float(a), float(b)) for a, b in st]))
    if normals is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, nn)) for nn in normals]))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if double:
        m.CreateDoubleSidedAttr(True)
    return m


def grid_mesh(path, fn, u_range, v_range, nu, nv, st_scale=1.0):
    """Parametric surface fn(u, v) -> (N,3)."""
    us, vs = np.linspace(*u_range, nu), np.linspace(*v_range, nv)
    U, V = np.meshgrid(us, vs, indexing="ij")
    P = fn(U, V).reshape(-1, 3)
    faces = []
    for i in range(nu - 1):
        for j in range(nv - 1):
            a = i * nv + j
            faces.append([a, a + nv, a + nv + 1, a + 1])
    st = np.stack([U.ravel(), V.ravel()], 1) * st_scale
    G = P.reshape(nu, nv, 3)
    du = np.gradient(G, axis=0)
    dv = np.gradient(G, axis=1)
    N = np.cross(du, dv)
    ln = np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.where(ln > 1e-9, N / np.maximum(ln, 1e-9), np.array([0, 0, 1.0]))
    out = G - G.reshape(-1, 3).mean(0)
    if np.sum(N * out) < 0:  # make normals point outward
        N = -N
    return mesh(path, P, faces, st=st, normals=N.reshape(-1, 3))


def torus_mesh(path, R, r, nu=48, nv=16):
    def f(U, V):
        return np.stack([(R + r * np.cos(V)) * np.cos(U), (R + r * np.cos(V)) * np.sin(U), r * np.sin(V)], -1)
    return grid_mesh(path, f, (0, 2 * math.pi), (0, 2 * math.pi), nu, nv)


def ellipsoid_mesh(path, a, b, c, nu=32, nv=16, bump=None, seed=0):
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, 6)

    def f(U, V):
        x, y, z = np.cos(V) * np.cos(U), np.cos(V) * np.sin(U), np.sin(V)
        k = 1.0
        if bump:
            k = 1 + bump * (0.5 * np.sin(3 * U + ph[0]) * np.cos(2 * V + ph[1]) + 0.3 * np.sin(5 * U + ph[2] + 3 * V)
                            + 0.2 * np.sin(9 * U + ph[3]) * np.sin(7 * V + ph[4]))
        return np.stack([a * x * k, b * y * k, c * z * k], -1)
    return grid_mesh(path, f, (0, 2 * math.pi), (-math.pi / 2, math.pi / 2), nu, nv)


# ============================================================ seabed
x0, x1, y0, y1 = TL.bounds
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
SB = 140.0


def seabed_h(X, Y):
    d = np.hypot(X - cx, Y - cy)
    dunes = 0.12 * np.sin(X * 0.23 + 0.7) * np.cos(Y * 0.19) + 0.06 * np.sin(X * 0.61 + Y * 0.37)
    far = np.clip((d - 30) / 40, 0, 1) ** 1.5 * (2.0 * np.sin(X * 0.07) * np.cos(Y * 0.05 + 1) + 1.5)
    return dunes + far


def build_seabed():
    n = 260
    xs = np.linspace(cx - SB / 2, cx + SB / 2, n)
    ys = np.linspace(cy - SB / 2, cy + SB / 2, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = seabed_h(X, Y)
    P = np.stack([X, Y, Z], -1).reshape(-1, 3)
    faces = [[i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1] for i in range(n - 1) for j in range(n - 1)]
    st = np.stack([X.ravel() / 4.0, Y.ravel() / 4.0], 1)
    m = mesh("/World/Seabed", P, faces, st=st)
    bind(m.GetPrim(), material("sand", rough=0.92, albedo_tex=f"{TEX}/sand_albedo.png",
                               normal_tex=f"{TEX}/sand_normal.png", emissive_tex=f"{TEX}/caustic_00.png"))


build_seabed()


# ============================================================ decor: rocks, kelp, sea grass
def build_decor():
    UsdGeom.Xform.Define(stage, "/World/Decor")
    rock_m = material("rockmat", color=(0.33, 0.31, 0.28), rough=0.85)
    moss_m = material("moss", color=(0.20, 0.24, 0.15), rough=0.9)
    kelp_m = material("kelp", color=(0.13, 0.22, 0.06), rough=0.5)
    objs = [np.array(o["xyz"][:2]) for o in TL.world["objects"]]
    path_pts = np.array([[s["pose"]["x"], s["pose"]["y"]] for s in TL.log])

    def clear(p, r):
        return all(np.linalg.norm(p - o) > r + 2.0 for o in objs) and np.min(np.linalg.norm(path_pts - p, axis=1)) > r + 2.5

    rocks, kelps = 0, 0
    tries = 0
    while (rocks < 60 or kelps < 34) and tries < 5000:
        tries += 1
        p = np.array([RNG.uniform(cx - 45, cx + 45), RNG.uniform(cy - 40, cy + 40)])
        if rocks < 60:
            r = RNG.uniform(0.3, 1.6) * (1.0 if RNG.random() > 0.15 else 2.2)
            if clear(p, r):
                z = float(seabed_h(np.array(p[0]), np.array(p[1]))) - 0.25 * r
                xf = xform(f"/World/Decor/rock{rocks}", (p[0], p[1], z), RNG.uniform(0, 360))
                ell = ellipsoid_mesh(f"/World/Decor/rock{rocks}/m", r, r * RNG.uniform(0.6, 1.0), r * RNG.uniform(0.45, 0.8),
                                     bump=0.25, seed=rocks)
                bind(ell.GetPrim(), rock_m if RNG.random() > 0.3 else moss_m)
                rocks += 1
                continue
        if kelps < 34 and clear(p, 1.5):
            z = float(seabed_h(np.array(p[0]), np.array(p[1])))
            g = xform(f"/World/Decor/kelp{kelps}", (p[0], p[1], z))
            for b in range(int(RNG.integers(6, 12))):
                hgt = RNG.uniform(2.0, 5.5)
                off = RNG.normal(0, 0.35, 2)
                ph = RNG.uniform(0, 6.28)

                def blade(U, V, hgt=hgt, off=off, ph=ph):
                    w = 0.05 * (1 - 0.7 * V)
                    sway = 0.35 * V ** 1.5 * np.sin(ph + 2.2 * V)
                    return np.stack([off[0] + U * w + sway, off[1] + 0.25 * V * np.cos(ph), V * hgt], -1)
                bm = grid_mesh(f"/World/Decor/kelp{kelps}/b{b}", blade, (-1, 1), (0, 1), 2, 24)
                bm.CreateDoubleSidedAttr(True)
                bind(bm.GetPrim(), kelp_m)
            kelps += 1
    log(f"decor: {rocks} rocks, {kelps} kelp clusters")


build_decor()


# ============================================================ mission objects
OBJ = {}


def build_objects():
    from sim import scene

    UsdGeom.Xform.Define(stage, "/World/Objects")
    rust_m = material("ordnance", rough=0.55, metal=0.35, albedo_tex=f"{TEX}/rust.png")
    mine_m = material("mine", color=(0.12, 0.13, 0.12), rough=0.45, metal=0.6, albedo_tex=f"{TEX}/rust.png")
    tyre_m = material("tyre", color=(0.035, 0.035, 0.035), rough=0.8)
    rock_m = material("rockmat", color=(0.33, 0.31, 0.28), rough=0.85)
    for o in TL.world["objects"]:
        path = scene.add_object(stage, o, parent_path="/World/Objects")
        prim = stage.GetPrimAtPath(path)
        m = {"aerial_bomb": rust_m, "sphere_mine": mine_m, "tyre": tyre_m, "rock": rock_m}[o["cls"]]
        if o["cls"] == "rock":  # nicer rock than the smooth proxy
            for c in prim.GetChildren():
                UsdGeom.Imageable(c).MakeInvisible()
            r = ellipsoid_mesh(path + "/Hero", 0.62, 0.46, 0.36, bump=0.3, seed=hash(o["id"]) % 100)
            UsdGeom.Xformable(r).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.28))
        if o["cls"] == "sphere_mine":  # contact horns
            for k in range(6):
                a, e = k * math.pi / 3, 0.5 if k % 2 else 0.95
                hz = UsdGeom.Cylinder.Define(stage, f"{path}/Horn{k}")
                hz.CreateRadiusAttr(0.035)
                hz.CreateHeightAttr(0.16)
                hz.CreateAxisAttr("Z")
                d = np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)])
                xf = UsdGeom.Xformable(hz)
                xf.AddTranslateOp().Set(Gf.Vec3d(*(d * 0.42 + np.array([0, 0, 0.375]))))
                rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*d))
                xf.AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
        bind(prim, m)
        z_final = o["xyz"][2]
        OBJ[o["id"]] = {"prim": prim, "xyz": o["xyz"], "cls": o["cls"]}
    log(f"objects: {len(OBJ)}; revealed by events: {TL.reveal_step}")


build_objects()


# ============================================================ AUV
def build_auv():
    root = xform("/World/AUV")
    body = xform("/World/AUV/Body")  # local +X = forward
    hull = material("hull", color=(0.98, 0.62, 0.05), rough=0.28, metal=0.1, clearcoat=1.0)
    black = material("hull_black", color=(0.02, 0.02, 0.025), rough=0.35, metal=0.2, clearcoat=0.6)
    metal = material("hull_metal", color=(0.55, 0.57, 0.6), rough=0.25, metal=1.0)
    glass = material("dome", color=(0.02, 0.05, 0.07), rough=0.05, metal=0.0, clearcoat=1.0)
    lamp = material("lamp", color=(1, 1, 1), emissive=(40.0, 46.0, 50.0))
    beacon = material("beacon", color=(0.2, 1, 0.3), emissive=(0.0, 30.0, 6.0))
    L, R = 2.1, 0.23

    def prof(U, V):  # torpedo body of revolution along X, U = x in [-1,1], V = angle
        x = U * L / 2
        t = (U + 1) / 2
        r = R * np.where(t > 0.86, np.sqrt(np.clip(1 - ((t - 0.86) / 0.14) ** 2, 0, 1)),
                         np.where(t < 0.18, 0.35 + 0.65 * np.sqrt(np.clip(t / 0.18, 0, 1)), 1.0))
        return np.stack([x, r * np.cos(V), r * np.sin(V)], -1)
    hm = grid_mesh("/World/AUV/Body/Hull", prof, (-1, 1), (0, 2 * math.pi), 60, 32)
    bind(hm.GetPrim(), hull)
    for k, xc in enumerate((0.35, -0.55)):
        band = UsdGeom.Cylinder.Define(stage, f"/World/AUV/Body/Band{k}")
        band.CreateAxisAttr("X"); band.CreateRadiusAttr(R * 1.012); band.CreateHeightAttr(0.07)
        UsdGeom.Xformable(band).AddTranslateOp().Set(Gf.Vec3d(xc, 0, 0))
        bind(band.GetPrim(), black)
    for k in range(4):  # tail fins
        a = k * 90 + 45
        fin = UsdGeom.Cube.Define(stage, f"/World/AUV/Body/Fin{k}")
        fx = UsdGeom.Xformable(fin)
        fx.AddRotateXOp().Set(float(a))
        fx.AddTranslateOp().Set(Gf.Vec3d(-0.78, 0, R + 0.1))
        fx.AddScaleOp().Set(Gf.Vec3f(0.17, 0.012, 0.11))
        bind(fin.GetPrim(), black)
    shroud = torus_mesh("/World/AUV/Body/Shroud", R * 0.95, 0.03)
    sx = UsdGeom.Xformable(shroud)
    sx.AddTranslateOp().Set(Gf.Vec3d(-1.12, 0, 0)); sx.AddRotateYOp().Set(90.0)
    bind(shroud.GetPrim(), metal)
    prop = xform("/World/AUV/Body/Prop", (-1.12, 0, 0))
    UsdGeom.Xformable(prop).AddRotateXOp().Set(0.0)
    for k in range(4):
        bl = UsdGeom.Cube.Define(stage, f"/World/AUV/Body/Prop/B{k}")
        bx = UsdGeom.Xformable(bl)
        bx.AddRotateXOp().Set(float(k * 90))
        bx.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.09))
        bx.AddRotateZOp().Set(25.0)
        bx.AddScaleOp().Set(Gf.Vec3f(0.02, 0.06, 0.16))
        bind(bl.GetPrim(), metal)
    dome = ellipsoid_mesh("/World/AUV/Body/SonarDome", 0.16, 0.12, 0.08)
    UsdGeom.Xformable(dome).AddTranslateOp().Set(Gf.Vec3d(0.78, 0, -R * 0.9))
    bind(dome.GetPrim(), glass)
    mast = UsdGeom.Cylinder.Define(stage, "/World/AUV/Body/Mast")
    mast.CreateRadiusAttr(0.015); mast.CreateHeightAttr(0.3)
    UsdGeom.Xformable(mast).AddTranslateOp().Set(Gf.Vec3d(-0.3, 0, R + 0.15))
    bind(mast.GetPrim(), black)
    bcn = UsdGeom.Sphere.Define(stage, "/World/AUV/Body/Beacon")
    bcn.CreateRadiusAttr(0.035)
    UsdGeom.Xformable(bcn).AddTranslateOp().Set(Gf.Vec3d(-0.3, 0, R + 0.32))
    bind(bcn.GetPrim(), beacon)
    for k, yy in enumerate((0.12, -0.12)):  # headlights
        lp = UsdGeom.Sphere.Define(stage, f"/World/AUV/Body/Lamp{k}")
        lp.CreateRadiusAttr(0.04)
        UsdGeom.Xformable(lp).AddTranslateOp().Set(Gf.Vec3d(0.98, yy, -0.06))
        bind(lp.GetPrim(), lamp)
        lt = UsdLux.SphereLight.Define(stage, f"/World/AUV/Body/Light{k}")
        lt.CreateRadiusAttr(0.03)
        lt.CreateIntensityAttr(1.6e5)
        lt.CreateColorAttr(Gf.Vec3f(0.92, 0.97, 1.0))
        sh = UsdLux.ShapingAPI.Apply(lt.GetPrim())
        sh.CreateShapingConeAngleAttr(24.0)
        sh.CreateShapingConeSoftnessAttr(0.35)
        lx = UsdGeom.Xformable(lt)
        lx.AddTranslateOp().Set(Gf.Vec3d(1.02, yy, -0.06))
        lx.AddRotateYOp().Set(-62.0)  # light shines along its -Z: this aims it forward and 28 deg down
    return root


AUV = build_auv()
AUV_PROP = UsdGeom.Xformable(stage.GetPrimAtPath("/World/AUV/Body/Prop"))


# ============================================================ sonar fan (translucent, sweeping)
def fan_mesh(path, half_deg, r0, r1, n=40):
    pts, faces = [], []
    for i in range(n + 1):
        a = math.radians(-half_deg + 2 * half_deg * i / n)
        pts += [(r0 * math.cos(a), r0 * math.sin(a), 0), (r1 * math.cos(a), r1 * math.sin(a), 0)]
    for i in range(n):
        faces.append([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2])
    return mesh(path, pts, faces, double=True)


def ribbon(path, pts, width=0.025):
    """Flat emissive line through 3D points (in the parent's XY plane)."""
    pts = np.asarray(pts, float)
    P, faces = [], []
    for i, p in enumerate(pts):
        d = pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]
        n = np.array([-d[1], d[0], 0.0]) / (np.linalg.norm(d[:2]) + 1e-9) * width / 2
        P += [p - n, p + n]
    for i in range(len(pts) - 1):
        faces.append([2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2])
    return mesh(path, P, faces, double=True)


FAN = xform("/World/AUV/Fan", (1.0, 0, -0.2))
UsdGeom.Xformable(FAN).AddRotateYOp().Set(25.0)  # tilt 25 deg down
FAN_M = material("fan", color=(0.1, 0.9, 0.8), emissive=(0.0, 2.2, 1.8), rough=1.0)
HALF, RMAX = math.radians(65), 10.0
for k, a_ in enumerate((-HALF, HALF)):
    bind(ribbon(f"/World/AUV/Fan/Edge{k}", [(0.3 * math.cos(a_), 0.3 * math.sin(a_), 0),
                                           (RMAX * math.cos(a_), RMAX * math.sin(a_), 0)]).GetPrim(), FAN_M)
for k, rr in enumerate((RMAX, RMAX * 0.5)):
    arc = [(rr * math.cos(t_), rr * math.sin(t_), 0) for t_ in np.linspace(-HALF, HALF, 60)]
    bind(ribbon(f"/World/AUV/Fan/Arc{k}", arc, width=0.03 if k == 0 else 0.015).GetPrim(), FAN_M)
SWEEP = xform("/World/AUV/Fan/Sweep")
bind(ribbon("/World/AUV/Fan/Sweep/S", [(0.3, 0, 0), (RMAX, 0, 0)], width=0.05).GetPrim(),
     material("sweep", color=(0.3, 1, 0.9), emissive=(0.0, 6.0, 4.5), rough=1.0))


# ============================================================ mission graphics
UsdGeom.Xform.Define(stage, "/World/HUD3D")
RED_RING = material("ring_red", color=(1, 0.1, 0.05), emissive=(9.0, 0.7, 0.2), rough=1.0)
RED_WIRE = material("wire_red", color=(1, 0.15, 0.08), emissive=(2.2, 0.18, 0.06), rough=1.0)
VCOL = {None: (1.0, 0.65, 0.05), "pending": (1.0, 0.65, 0.05), "UXO": (1.0, 0.08, 0.05),
        "uncertain": (1.0, 0.85, 0.05), "non-UXO": (0.1, 1.0, 0.35)}
BEAM = {k: material(f"beam_{i}", color=c, emissive=tuple(4 * x for x in c), rough=1.0)
        for i, (k, c) in enumerate(VCOL.items()) if k is not None}
ZONES, BEACONS = {}, {}


def zone(cid, xy, r):
    if cid in ZONES:
        return ZONES[cid]
    g = xform(f"/World/HUD3D/Zone_{cid}", (xy[0], xy[1], 0.0))
    ring = torus_mesh(f"/World/HUD3D/Zone_{cid}/Ring", r, 0.04, nu=128, nv=8)
    bind(ring.GetPrim(), RED_RING)
    UsdGeom.Xformable(ring).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.12))
    for k, th in enumerate((0.45, 1.0)):  # latitude rings of a squashed dome (height 0.55 r)
        rr, hz = r * math.cos(th), r * 0.55 * math.sin(th)
        lat = torus_mesh(f"/World/HUD3D/Zone_{cid}/Lat{k}", rr, 0.007, nu=96, nv=6)
        UsdGeom.Xformable(lat).AddTranslateOp().Set(Gf.Vec3d(0, 0, hz + 0.05))
        bind(lat.GetPrim(), RED_WIRE)
    for k in range(12):  # meridians
        ph = 2 * math.pi * k / 12
        ths = np.linspace(0, math.pi / 2, 24)
        ctr = np.stack([r * np.cos(ths) * math.cos(ph), r * np.cos(ths) * math.sin(ph), r * 0.55 * np.sin(ths) + 0.05], 1)
        side = np.array([-math.sin(ph), math.cos(ph), 0.0]) * 0.007
        P = np.vstack([np.stack([c - side, c + side]) for c in ctr])
        faces = [[2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2] for i in range(len(ctr) - 1)]
        bind(mesh(f"/World/HUD3D/Zone_{cid}/Mer{k}", P, faces, double=True).GetPrim(), RED_WIRE)
    ZONES[cid] = g
    return g


def beacon(cid, xy):
    if cid in BEACONS:
        return BEACONS[cid]
    g = xform(f"/World/HUD3D/Beacon_{cid}", (xy[0], xy[1], 0.0))
    beam = UsdGeom.Cylinder.Define(stage, f"/World/HUD3D/Beacon_{cid}/Beam")
    beam.CreateRadiusAttr(0.012); beam.CreateHeightAttr(5.0)
    UsdGeom.Xformable(beam).AddTranslateOp().Set(Gf.Vec3d(0, 0, 3.2))
    halo = xform(f"/World/HUD3D/Beacon_{cid}/Halo", (0, 0, 1.6))
    tor = torus_mesh(f"/World/HUD3D/Beacon_{cid}/Halo/T", 0.5, 0.018, nu=64, nv=8)
    BEACONS[cid] = {"g": g, "beam": beam, "halo": halo, "tor": tor, "state": None}
    return BEACONS[cid]


ROUTE_N = 90
route_proto = "/World/HUD3D/RouteProto"
UsdGeom.Scope.Define(stage, route_proto)
dot = UsdGeom.Cylinder.Define(stage, route_proto + "/dot")
dot.CreateRadiusAttr(0.09); dot.CreateHeightAttr(0.02)
bind(dot.GetPrim(), material("route", color=(0.1, 0.9, 1.0), emissive=(0.4, 6.0, 9.0), rough=1.0))
old_dot = UsdGeom.Cylinder.Define(stage, route_proto + "/old")
old_dot.CreateRadiusAttr(0.07); old_dot.CreateHeightAttr(0.02)
bind(old_dot.GetPrim(), material("route_old", color=(1, 0.5, 0.1), emissive=(4.0, 1.2, 0.2), rough=1.0))
ROUTE = UsdGeom.PointInstancer.Define(stage, "/World/HUD3D/Route")
ROUTE.CreatePrototypesRel().SetTargets([dot.GetPath(), old_dot.GetPath()])


def route_points(path, spacing=0.55, max_len=40.0):
    pts, acc = [], 0.0
    for a, b in zip(path[:-1], path[1:]):
        a, b = np.array(a, float), np.array(b, float)
        seg = np.linalg.norm(b - a)
        k = 0.0
        while k < seg and acc < max_len:
            pts.append(a + (b - a) * (k / max(seg, 1e-6)))
            k += spacing
            acc += spacing
    return pts


# ============================================================ life: fish, marine snow, bubbles
def fish_mesh(path, length, color, seed):
    def f(U, V):
        t = (U + 1) / 2  # tail 0 -> head 1
        w = 0.5 * length * 0.22 * np.sin(np.pi * np.clip(t * 1.05, 0, 1)) ** 0.8
        h = w * 1.6
        return np.stack([U * length / 2, w * np.cos(V), h * np.sin(V)], -1)
    body = grid_mesh(path + "/body", f, (-1, 1), (0, 2 * math.pi), 18, 10)
    tail = mesh(path + "/tail", [(-length / 2 + 0.02, 0, 0), (-length / 2 - length * 0.22, 0, length * 0.16),
                                 (-length / 2 - length * 0.22, 0, -length * 0.16)], [[0, 1, 2]], double=True)
    m = material(f"fish{seed}", color=color, rough=0.3, metal=0.75)
    bind(body.GetPrim(), m)
    bind(tail.GetPrim(), m)


UsdGeom.Scope.Define(stage, "/World/Life/Protos")
fish_mesh("/World/Life/Protos/small", 0.34, (0.62, 0.70, 0.78), 0)
fish_mesh("/World/Life/Protos/yellow", 0.30, (1.0, 0.78, 0.10), 1)
fish_mesh("/World/Life/Protos/big", 0.9, (0.30, 0.34, 0.30), 2)
FISH = UsdGeom.PointInstancer.Define(stage, "/World/Life/Fish")
FISH.CreatePrototypesRel().SetTargets([Sdf.Path("/World/Life/Protos/small"), Sdf.Path("/World/Life/Protos/yellow"),
                                       Sdf.Path("/World/Life/Protos/big")])
SCHOOLS = []
for k in range(4):
    nf = int(RNG.integers(35, 60))
    SCHOOLS.append({"n": nf, "proto": 0 if k % 2 == 0 else 1, "R": RNG.uniform(10, 22), "h": RNG.uniform(2.8, 5.5),
                    "w": RNG.uniform(0.05, 0.09) * (1 if k % 2 else -1), "ph": RNG.uniform(0, 6.28),
                    "c": np.array([cx, cy]) + RNG.normal(0, 6, 2),
                    "off": RNG.normal(0, 1, (nf, 3)) * np.array([1.6, 1.6, 0.6]), "jit": RNG.uniform(0, 6.28, nf)})
BIG = [{"R": RNG.uniform(8, 18), "h": RNG.uniform(1.0, 3.0), "w": RNG.uniform(0.03, 0.05), "ph": RNG.uniform(0, 6.28),
        "c": np.array([cx, cy]) + RNG.normal(0, 8, 2)} for _ in range(5)]

UsdGeom.Scope.Define(stage, "/World/Life/SnowProto")
sp = UsdGeom.Sphere.Define(stage, "/World/Life/SnowProto/p")
sp.CreateRadiusAttr(0.012)
bind(sp.GetPrim(), material("snow", color=(0.9, 0.95, 0.9), emissive=(0.35, 0.42, 0.40), rough=1.0))
bp = UsdGeom.Sphere.Define(stage, "/World/Life/SnowProto/bubble")
bp.CreateRadiusAttr(0.013)
bind(bp.GetPrim(), material("bubble", color=(0.85, 0.95, 1.0), emissive=(0.15, 0.2, 0.22), rough=0.05, metal=0.6))
SNOW = UsdGeom.PointInstancer.Define(stage, "/World/Life/Snow")
SNOW.CreatePrototypesRel().SetTargets([sp.GetPath(), bp.GetPath()])
NS = 3500
SNOW0 = RNG.uniform([-14, -14, 0.2], [14, 14, 9], (NS, 3))
SNOWV = RNG.uniform(0.02, 0.07, NS)


def quat_yaw_pitch(yaw, pitch=0.0):
    r = Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)) * Gf.Rotation(Gf.Vec3d(0, 1, 0), -math.degrees(pitch))
    q = r.GetQuat()
    return Gf.Quath(float(q.GetReal()), *map(float, q.GetImaginary()))


def update_life(f):
    t = f / FPS
    pos, ori, proto = [], [], []
    for s in SCHOOLS:
        a = s["ph"] + s["w"] * t * 6
        c = s["c"] + s["R"] * np.array([math.cos(a), math.sin(a) * 0.7])
        vel = np.array([-math.sin(a), math.cos(a) * 0.7]) * np.sign(s["w"])
        yaw = math.atan2(vel[1], vel[0])
        for i in range(s["n"]):
            o = s["off"][i]
            wig = 0.25 * math.sin(t * 2.3 + s["jit"][i])
            p = (c[0] + o[0] + wig * math.cos(yaw + 1.57), c[1] + o[1] + wig * math.sin(yaw + 1.57),
                 s["h"] + o[2] + 0.15 * math.sin(t * 1.7 + s["jit"][i]))
            pos.append(Gf.Vec3f(*p))
            ori.append(quat_yaw_pitch(yaw + 0.25 * math.sin(t * 6 + s["jit"][i])))
            proto.append(s["proto"])
    for b in BIG:
        a = b["ph"] + b["w"] * t * 6
        p = b["c"] + b["R"] * np.array([math.cos(a), math.sin(a)])
        pos.append(Gf.Vec3f(float(p[0]), float(p[1]), b["h"] + 0.3 * math.sin(t * 0.7)))
        ori.append(quat_yaw_pitch(a + math.pi / 2 + 0.15 * math.sin(t * 3)))
        proto.append(2)
    FISH.GetPositionsAttr().Set(Vt.Vec3fArray(pos))
    FISH.GetOrientationsAttr().Set(Vt.QuathArray(ori))
    FISH.GetProtoIndicesAttr().Set(Vt.IntArray(proto))
    # marine snow follows the camera volume (wraps), plus prop bubbles
    c = TL.cam_pos[f]
    sp_ = SNOW0.copy()
    sp_[:, 2] -= SNOWV * t
    sp_[:, 0] += 0.05 * np.sin(t * 0.5 + SNOW0[:, 1])
    rel = sp_ - np.array([c[0], c[1], 0.0])
    rel[:, :2] = (rel[:, :2] + 14) % 28 - 14
    rel[:, 2] = (rel[:, 2] - 0.2) % 8.8 + 0.2
    world = rel + np.array([c[0], c[1], 0.0])
    bub = []
    for k in range(70):  # bubbles shed from the prop over the last ~3 s
        g = max(f - k * 2, 0)
        yaw = TL.robot_yaw[g]
        if TL.robot_speed[g] < 0.02 and k > 2:
            continue
        p = TL.robot_pos[g] - 1.15 * np.array([math.cos(yaw), math.sin(yaw), 0])
        age = k * 2 / FPS
        jitter = np.array([math.sin(k * 12.9898) * 0.12, math.sin(k * 78.233) * 0.12, math.sin(k * 3.7) * 0.05])
        bub.append(p + jitter + np.array([0, 0, 0.35 * age]))
    allp = np.vstack([world, np.array(bub)]) if bub else world
    SNOW.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in allp]))
    SNOW.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(world) + [1] * len(bub)))


# ============================================================ lights
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1500.0)
sun.CreateAngleAttr(2.5)
sun.CreateColorAttr(Gf.Vec3f(0.75, 0.95, 1.0))
sx = UsdGeom.Xformable(sun)
sx.AddRotateXYZOp().Set(Gf.Vec3f(18.0, -14.0, 30.0))
dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(180.0)
dome.CreateColorAttr(Gf.Vec3f(0.08, 0.35, 0.50))


# ============================================================ camera
cam = UsdGeom.Camera.Define(stage, "/World/BeautyCam")
cam.CreateHorizontalApertureAttr(36.0)
cam.CreateVerticalApertureAttr(36.0 * H / W)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 400.0))
cam.CreateFStopAttr(0.0)  # no depth of field (keeps the action readable)
CAMX = UsdGeom.Xformable(cam)
CAM_OP = CAMX.AddTransformOp()


def set_camera(f):
    c, t = TL.cam_pos[f], TL.cam_tgt[f]
    fwd = (t - c) / (np.linalg.norm(t - c) + 1e-9)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    back = -fwd
    m = Gf.Matrix4d(right[0], right[1], right[2], 0, up[0], up[1], up[2], 0, back[0], back[1], back[2], 0,
                    c[0], c[1], c[2], 1)
    CAM_OP.Set(m)
    cam.GetFocalLengthAttr().Set(float(18.0 / math.tan(math.radians(TL.cam_fov[f]) / 2)))


# ============================================================ per-frame update
def _ops(x):
    x = x if isinstance(x, UsdGeom.Xformable) else UsdGeom.Xformable(x)
    return {op.GetOpName(): op for op in x.GetOrderedXformOps()}


def previous_plan(step, linger=8):
    """The route that was planned before the most recent re-plan, shown for `linger` steps after it."""
    k = step
    while k >= 0 and TL.log[k]["state"] != "REPLAN":
        k -= 1
        if step - k > linger:
            return []
    if k < 0:
        return []
    j = k
    while j >= 0 and TL.log[j]["state"] in ("REPLAN", "HOLD", "BACKOFF"):
        j -= 1
    return TL.log[j].get("planned_path") or [] if j >= 0 else []


_last_murk = [-1.0]


def update_scene(f):
    t = f / FPS
    e = TL.entry(f)
    step = TL.step_of(f)
    # AUV
    p, yaw = TL.robot_pos[f], TL.robot_yaw[f]
    bob = 0.03 * math.sin(t * 1.3)
    set_xf(AUV, t=(p[0], p[1], p[2] + bob), rz=math.degrees(yaw))
    _ops(AUV_PROP)["xformOp:rotateX"].Set(float((t * 900) % 360) if TL.robot_speed[f] > 0.02 else float((t * 120) % 360))
    # sonar sweep
    sw = 63 * math.sin(t * 2.4)
    _ops(SWEEP)["xformOp:rotateZ"].Set(float(sw))
    # objects: hidden until revealed, then rise out of the sand
    for oid, o in OBJ.items():
        pr = TL.reveal_progress(oid, f)
        x, y, z = o["xyz"]
        if pr < 0:
            UsdGeom.Imageable(o["prim"]).MakeInvisible()
        else:
            UsdGeom.Imageable(o["prim"]).MakeVisible()
            k = 1 - (1 - pr) ** 3
            UsdGeom.Xformable(o["prim"]).GetOrderedXformOps()[0].Set(Gf.Vec3d(x, y, z - 0.7 * (1 - k)))
    # keep-out zones (pulse) and contact beacons
    live = {z_["id"]: z_ for z_ in e.get("keep_out") or []}
    for cid, g in list(ZONES.items()):
        UsdGeom.Imageable(g).MakeInvisible()
    for cid, z_ in live.items():
        g = zone(cid, z_["xy"], z_["r"])
        UsdGeom.Imageable(g).MakeVisible()
        pulse = 1.0 + 0.04 * math.sin(t * 4.0)
        set_xf(g, t=(z_["xy"][0], z_["xy"][1], 0.0), s=(pulse, pulse, 1.0))
    for c in e.get("contacts") or []:
        b = beacon(c["id"], c["xy"])
        v = TL.verdict_at(c["id"], f) or "pending"
        if b["state"] != v:
            bind(b["beam"].GetPrim(), BEAM[v])
            bind(b["tor"].GetPrim(), BEAM[v])
            b["state"] = v
        set_xf(b["halo"], rz=(t * 60) % 360, t=(0, 0, 1.6 + 0.12 * math.sin(t * 2)))
    # route trail
    path = e.get("planned_path") or []
    pts = route_points([[p[0], p[1]]] + path) if path else []
    prev = previous_plan(step)
    old = route_points(prev, spacing=0.9, max_len=30) if prev and prev != path else []
    allp = [(q[0], q[1], float(seabed_h(np.array(q[0]), np.array(q[1]))) + 0.06) for q in pts] + \
           [(q[0], q[1], float(seabed_h(np.array(q[0]), np.array(q[1]))) + 0.05) for q in old]
    ROUTE.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in allp]))
    ROUTE.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(pts) + [1] * len(old)))
    # caustics animation
    et = _mats.get("sand:emi")
    if et is not None:
        et.GetInput("file").Set(f"{TEX}/caustic_{(f // 2) % 24:02d}.png")
    # murk
    mk = TL.murk(f)
    if abs(mk - _last_murk[0]) > 0.02:
        apply_render_settings(args.mode, mk)
        _last_murk[0] = mk
    update_life(f)
    set_camera(f)


# ============================================================ capture
rp = rep.create.render_product(str(cam.GetPath()), (W, H))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach([rp])
try:
    rep.orchestrator.set_capture_on_play(False)
except Exception:  # noqa: BLE001
    pass

if args.frames:
    frames = [int(x) for x in args.frames.split(",")]
else:
    end = TL.n_frames if args.end < 0 else min(args.end, TL.n_frames)
    frames = list(range(args.start, end))
log(f"run {TL.run_id}: {TL.n_steps} steps -> {TL.n_frames} frames; rendering {len(frames)} "
    f"({frames[0]}..{frames[-1]}) at {W}x{H}, mode {args.mode}, {args.spp} subframes")

# warm-up (shader compile) on the first frame
update_scene(frames[0])
for _ in range(3):
    rep.orchestrator.step(rt_subframes=4)
log("warm-up done")

modes = args.variants.split(",") if args.variants else [args.mode]
for i, f in enumerate(frames):
    t0 = time.time()
    update_scene(f)
    for mode in modes:
        if len(modes) > 1:
            apply_render_settings(mode, TL.murk(f))
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=2)
        rep.orchestrator.step(rt_subframes=args.spp)
        img = ann.get_data()
        if img is None or not hasattr(img, "shape") or img.size == 0:
            log("empty frame", f)
            continue
        name = f"{f:05d}.png" if len(modes) == 1 else f"{f:05d}_{mode}.png"
        Image.fromarray(np.asarray(img)[:, :, :3]).save(os.path.join(args.out, name))
    if i % 10 == 0 or len(frames) < 20:
        log(f"frame {f} ({i + 1}/{len(frames)}) {time.time() - t0:.1f}s shot={TL.shots[f]} state={TL.entry(f)['state']}")

log("done")
app.close()
