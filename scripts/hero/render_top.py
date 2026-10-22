"""Render a car model from directly above into the maps the menu hero lights (run in Blender).

    blender -b --factory-startup -P scripts/hero/render_top.py -- model.glb out_dir \
        [--preview] [--blackout [--finish gloss|metallic|matte]] [--stage [--stage-test]]
        [--drive [--stage-test]]

Imports the model, puts its nose towards the top of the image, and renders orthographic
top-down images with Cycles:

- ``beauty.png``: a studio-lit look (for checking; ``--preview`` renders only this).
- ``albedo.png``: each material's base colour, unlit.
- ``normal.png``: the surface direction (normal maps included), ``0.5 + 0.5 n``, in image
  axes: x right, y up the image (towards the nose), z towards the camera.
- ``surface.png``: R roughness, G metallic, B height above the ground / ``HEIGHT_SCALE_M``.
- ``ao.png``: R ambient occlusion (crevices darker), G clear-coat weight.

``albedo`` is sRGB-encoded; the other passes are raw values. All are RGBA with the car's
coverage in alpha, ``PX_PER_M`` pixels per metre, the car centred
across and the nose ``MARGIN_M`` below the top edge. The unlit passes replace each material's
surface with an emission shader carrying the value (and the material's alpha, so decals and
cut-outs keep their shape); nothing needs Blender's compositor. Decal materials (names in
``DECALS``) are deleted: the logos go, the paint stays. ``SURFACES`` gives the main materials
realistic roughness, metal and clear coat (the model's own values are flat).

``--blackout`` paints the whole car black: the paint becomes solid gloss black, and every
other material is turned grey and darkened by ``BLACKOUT_LEVELS`` (carbon weave, tyre and
metal detail survive as shades of black, but no colour does). ``--finish`` then gives the
whole body (paint and carbon) one finish from ``FINISHES``: ``metallic`` (the default: dark
metal flake under a clear coat), ``gloss`` (gloss
black under a clear coat) or ``matte``
(satin black, no coat). Tyres stay rubber and metal parts stay metal.

**Light stage** (``--stage``): instead of the passes above, photograph the front of the car
(``STAGE_LENGTH_M`` behind the nose) path-traced, once per position of a softbox
(``STAGE_SIZE_M`` disc, ``STAGE_LIGHT_Z_M`` above the ground) on the grid ``STAGE_XS`` x
``STAGE_YS`` over the car, into ``out_dir/stage/light_<row>_<col>.png``, plus
``ambient.png`` (a dim, broad overhead light: the car at rest). The menu blends the four
photos nearest the cursor, so every light position has real soft shadows, reflections of the
softbox in the clear coat and occlusion. ``stage.json`` records the grid (positions in map
metres: x right of the centre line, y down from the top edge). ``--stage-test`` renders three
positions only, for tuning.

**Driving stage** (``--drive``): the light stage for the driving hero
(:mod:`f1sim.app.ui.hero_drive`). The whole car, at ``DRIVE_PX_PER_M``, on a softbox grid
``STAGE_XS`` x ``DRIVE_YS`` over its full length, in two layers so the front wheels can
steer: ``drive_stage/body_<row>_<col>.png`` (everything but the front wheels) and
``wheel_<row>_<col>.png`` (only the front wheels), plus ``body_ambient`` and
``wheel_ambient``. The front wheels are found by material (tyres, rims, discs) at the axle
nearer the nose; then every face inside their volume (rim covers, brake ducts, upright)
goes with them. In ``body_ambient`` a shadow-catcher ground plane
puts the car's soft contact shadow (not the front wheels') into the alpha, so it falls on the
track drawn below; the softbox photos have no ground.
``stage.json`` adds the wheel centres and both axles, in map metres.

``meta.json`` records the scale, the image size and the model's bounds.
"""

import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Euler, Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
MODEL, OUT = str(Path(argv[0]).resolve()), Path(argv[1]).resolve()
PREVIEW = "--preview" in argv
BLACKOUT = "--blackout" in argv
FINISH = argv[argv.index("--finish") + 1] if "--finish" in argv else "metallic"
STAGE = "--stage" in argv
STAGE_TEST = "--stage-test" in argv
DRIVE = "--drive" in argv
OUT.mkdir(parents=True, exist_ok=True)

PX_PER_M = 700
MARGIN_M = 0.10
HEIGHT_SCALE_M = 1.5
DECALS = ("sf25_decals", "sf25_decals2", "numbers", "sf25_sw_decals")
RECOLOUR = {"gp21_cinture": (0.025, 0.025, 0.028, 1.0)}
"""Materials whose texture carries printed text, painted one plain colour instead."""
BLACKOUT_PAINT = (
    "sf25_paints",
    "sf25_details",
    "sf25_generics",
    "driver_color",
    "kers_red",
    "kers_blue",
    "kers_green",
    "rear_light",
    "mirrorlight",
    "CLARED",
    "CLEARLED",
)
"""Painted (or coloured-light) materials that become solid gloss black with ``--blackout``."""
BLACKOUT_PAINT_COLOUR = (0.012, 0.012, 0.013, 1.0)
BLACKOUT_LEVELS = {"tyre_tread": 0.12, "tyre_sides": 0.12}
"""Greyscale multiplier per material with ``--blackout`` (default ``BLACKOUT_DEFAULT``)."""
BLACKOUT_DEFAULT = 0.12
BODYWORK = (
    "sf25_paints",
    "sf25_details",
    "sf25_generics",
    "sf25_carbon1",
    "sf25_carbon1_white",
    "sf25_carbon2",
    "sf25_carbon3",
)
"""Materials that make up the body: they all take the ``--finish``."""
FINISHES = {
    # name: (base colour, roughness, metallic, clear coat)
    "gloss": (0.012, 0.14, 0.0, 1.0),
    "metallic": (0.02, 0.30, 0.85, 1.0),
    "matte": (0.025, 0.62, 0.0, 0.0),
}
"""Body finishes for ``--blackout``. Metallic paint is approximated as a dark metal (the
flake) under a clear coat; matte is a rough dielectric with no coat."""
# Realistic surface values by material (the game conversion left roughness ~0.6 and no metal
# everywhere): (roughness, metallic, clear coat). Matched on the name before any ".001".
SURFACES = {
    "sf25_paints": (0.14, 0.0, 1.0),
    "sf25_details": (0.14, 0.0, 1.0),
    "sf25_generics": (0.20, 0.0, 0.8),
    "sf25_carbon1": (0.32, 0.0, 0.7),
    "sf25_carbon1_white": (0.32, 0.0, 0.7),
    "sf25_carbon2": (0.30, 0.0, 0.7),
    "sf25_carbon3": (0.45, 0.0, 0.0),
    "chrome": (0.18, 1.0, 0.0),
    "chrome_ao": (0.22, 1.0, 0.0),
    "discs": (0.35, 1.0, 0.0),
    "sf25_wheel_hub": (0.30, 0.8, 0.0),
    "glass": (0.05, 0.0, 0.0),
    "sf25_mirrorl": (0.05, 1.0, 0.0),
    "tyre_tread": (0.85, 0.0, 0.0),
    "tyre_sides": (0.75, 0.0, 0.0),
    "black_s": (0.7, 0.0, 0.0),
    "plastic_interior": (0.6, 0.0, 0.0),
}
SAMPLES_BEAUTY = 256
SAMPLES_DATA = 16
STAGE_PX_PER_M = 320
STAGE_LENGTH_M = 4.15
STAGE_XS = (-0.9, -0.45, 0.0, 0.45, 0.9)
STAGE_YS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
"""Softbox positions, map metres (x right of the centre line, y down from the image top)."""
STAGE_LIGHT_Z_M = 1.9
STAGE_SIZE_M = 0.7
STAGE_POWER_W = 300.0
STAGE_AMBIENT_POWER_W = 60.0
SAMPLES_STAGE = 96
DRIVE_PX_PER_M = 240
DRIVE_YS = tuple(round(0.55 * i, 2) for i in range(11))
"""Driving-stage softbox rows over the full car, metres down from the image top."""
WHEEL_MATS = ("tyre_tread", "tyre_sides", "sf25_wheel_hub", "discs")
WHEEL_HALF_WIDTH_M = 0.16
WHEEL_GRAB_M = 0.04
"""Faces within a front wheel's box (half its tread width and its radius, plus this) go with
the wheel, whatever their material."""

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=MODEL)
scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH"]


def bounds():
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    return (
        Vector([min(p[i] for p in pts) for i in range(3)]),
        Vector([max(p[i] for p in pts) for i in range(3)]),
    )


# ---------------------------------------------------------------- orientation and placement

lo, hi = bounds()
along = 0 if (hi.x - lo.x) > (hi.y - lo.y) else 1  # the long axis


def end_height(at_max: bool) -> float:
    """Highest point within 0.5 m of one end of the long axis (the rear wing is tall)."""
    edge = hi[along] if at_max else lo[along]
    top = 0.0
    for o in meshes:
        mw = o.matrix_world
        for v in o.data.vertices:
            p = mw @ v.co
            if abs(p[along] - edge) < 0.5:
                top = max(top, p.z)
    return top


rear_at_max = end_height(True) > end_height(False)
# Rotate about z so the nose points along +y (the top of the image)
nose_dir = Vector((0, 0, 0))
nose_dir[along] = -1.0 if rear_at_max else 1.0
angle = math.atan2(nose_dir.y, nose_dir.x)  # current nose heading
rot = Euler((0.0, 0.0, math.pi / 2 - angle))
for o in meshes:
    if o.parent is None:
        o.matrix_world = rot.to_matrix().to_4x4() @ o.matrix_world
bpy.context.view_layer.update()
lo, hi = bounds()
# Centre across, ground at z = 0, nose at y = 0 (the rear towards -y)
shift = Vector((-(lo.x + hi.x) / 2, -hi.y, -lo.z))
for o in meshes:
    if o.parent is None:
        o.location += shift
bpy.context.view_layer.update()
lo, hi = bounds()
width_m = hi.x - lo.x
length_m = hi.y - lo.y
img_w_m = width_m + 2 * MARGIN_M
img_h_m = length_m + 2 * MARGIN_M

# ----------------------------------------------------------------------- camera and render

cam_data = bpy.data.cameras.new("top")
cam_data.type = "ORTHO"
cam_data.ortho_scale = max(img_w_m, img_h_m)
cam = bpy.data.objects.new("top", cam_data)
scene.collection.objects.link(cam)
cam.location = (0.0, -length_m / 2, hi.z + 5.0)
cam.rotation_euler = (0.0, 0.0, 0.0)  # looking straight down -z, image up = +y
scene.camera = cam
cam_data.clip_end = 20.0

r = scene.render
r.engine = "CYCLES"
r.resolution_x = int(round(img_w_m * PX_PER_M))
r.resolution_y = int(round(img_h_m * PX_PER_M))
r.resolution_percentage = 100
r.film_transparent = True
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGBA"
r.image_settings.color_depth = "16"
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
cyc = scene.cycles
cyc.device = "GPU"
prefs = bpy.context.preferences.addons["cycles"].preferences
for kind in ("OPTIX", "CUDA", "NONE"):
    try:
        prefs.compute_device_type = kind
        break
    except TypeError:
        continue
prefs.get_devices()
for d in prefs.devices:
    d.use = True
cyc.use_denoising = True

# ------------------------------------------------------------------------------ materials

orig = {}


def principled(mat):
    if not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return n
    return None


def source(sock):
    """The socket feeding ``sock``, or None if it's unlinked."""
    return sock.links[0].from_socket if sock.is_linked else None


for mat in bpy.data.materials:
    p = principled(mat)
    if p is None:
        continue
    out = next(n for n in mat.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
    orig[mat.name] = {
        "out": out,
        "surface": source(out.inputs["Surface"]),
        "p": p,
    }


def recolour():
    for mat in bpy.data.materials:
        colour = RECOLOUR.get(mat.name.split(".")[0])
        if colour is None or mat.name not in orig:
            continue
        p = orig[mat.name]["p"]
        for link in list(p.inputs["Base Color"].links):
            mat.node_tree.links.remove(link)
        p.inputs["Base Color"].default_value = colour


def blackout():
    """Black every material: paint solid gloss black, the rest grey and darkened."""
    for mat in bpy.data.materials:
        if mat.name not in orig:
            continue
        key = mat.name.split(".")[0]
        p = orig[mat.name]["p"]
        nt = mat.node_tree
        sock = p.inputs["Base Color"]
        if key in BLACKOUT_PAINT or not sock.is_linked:
            for link in list(sock.links):
                nt.links.remove(link)
            if key in BLACKOUT_PAINT:
                sock.default_value = BLACKOUT_PAINT_COLOUR
            else:
                r, g, b, a = sock.default_value
                y = (0.2126 * r + 0.7152 * g + 0.0722 * b) * BLACKOUT_DEFAULT
                sock.default_value = (y, y, y, a)
        else:
            src = source(sock)
            bw = nt.nodes.new("ShaderNodeRGBToBW")
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = BLACKOUT_LEVELS.get(key, BLACKOUT_DEFAULT)
            nt.links.new(src, bw.inputs[0])
            nt.links.new(bw.outputs[0], mul.inputs[0])
            nt.links.new(mul.outputs[0], sock)
        # No glowing colour either (lights are emissive in some models)
        em = p.inputs.get("Emission Strength")
        if em is not None and not em.is_linked:
            em.default_value = 0.0


def apply_finish(name: str) -> None:
    """Give every bodywork material the finish ``name`` (see ``FINISHES``)."""
    if name not in FINISHES:
        raise SystemExit(f"--finish must be one of {', '.join(FINISHES)}, got {name!r}")
    base, rough, metal, coat = FINISHES[name]
    for mat in bpy.data.materials:
        key = mat.name.split(".")[0]
        if key not in BODYWORK or mat.name not in orig:
            continue
        p = orig[mat.name]["p"]
        nt = mat.node_tree
        if key in BLACKOUT_PAINT:  # solid paint; the carbon keeps its (greyscale) weave
            for link in list(p.inputs["Base Color"].links):
                nt.links.remove(link)
            p.inputs["Base Color"].default_value = (base, base, base * 1.05, 1.0)
        for sock_name, v in (("Roughness", rough), ("Metallic", metal), ("Coat Weight", coat)):
            sock = p.inputs.get(sock_name)
            if sock is None:
                continue
            for link in list(sock.links):
                nt.links.remove(link)
            sock.default_value = v


def delete_decals():
    """Remove the logo geometry (decal overlays): hidden-but-present overlays would still
    block ambient-occlusion rays and darken the paint under them."""
    import bmesh

    for o in meshes:
        slots = {
            i
            for i, s in enumerate(o.material_slots)
            if s.material and s.material.name.split(".")[0] in DECALS
        }
        if not slots:
            continue
        bm = bmesh.new()
        bm.from_mesh(o.data)
        doomed = [f for f in bm.faces if f.material_index in slots]
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
        bm.to_mesh(o.data)
        bm.free()
        print("DECAL_FACES_REMOVED", o.name, len(doomed), flush=True)


def set_surfaces():
    for mat in bpy.data.materials:
        vals = SURFACES.get(mat.name.split(".")[0])
        if vals is None or mat.name not in orig:
            continue
        p = orig[mat.name]["p"]
        for name, v in zip(("Roughness", "Metallic", "Coat Weight"), vals, strict=True):
            sock = p.inputs.get(name)
            if sock is None:
                continue
            for link in list(sock.links):
                mat.node_tree.links.remove(link)
            sock.default_value = v
        coat_r = p.inputs.get("Coat Roughness")
        if coat_r is not None:
            coat_r.default_value = 0.04


def set_emission(kind: str) -> None:
    """Replace every material's surface by an emission of one pass (keeping its alpha)."""
    for mat in bpy.data.materials:
        if mat.name not in orig:
            continue
        nt = mat.node_tree
        p, out = orig[mat.name]["p"], orig[mat.name]["out"]
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        if kind == "albedo":
            s = source(p.inputs["Base Color"])
            if s is not None:
                nt.links.new(s, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = p.inputs["Base Color"].default_value
        elif kind == "normal":
            s = source(p.inputs["Normal"])
            if s is None:
                geo = nt.nodes.new("ShaderNodeNewGeometry")
                s = geo.outputs["Normal"]
            # World normal -> 0..1 (the camera looks down -z, so image axes = world x, y)
            sc = nt.nodes.new("ShaderNodeVectorMath")
            sc.operation = "MULTIPLY_ADD"
            sc.inputs[1].default_value = (0.5, 0.5, 0.5)
            sc.inputs[2].default_value = (0.5, 0.5, 0.5)
            nt.links.new(s, sc.inputs[0])
            nt.links.new(sc.outputs[0], em.inputs["Color"])
        elif kind == "surface":
            comb = nt.nodes.new("ShaderNodeCombineColor")
            for i, name in ((0, "Roughness"), (1, "Metallic")):
                s = source(p.inputs[name])
                if s is not None:
                    nt.links.new(s, comb.inputs[i])
                else:
                    comb.inputs[i].default_value = p.inputs[name].default_value
            geo = nt.nodes.new("ShaderNodeNewGeometry")
            sep = nt.nodes.new("ShaderNodeSeparateXYZ")
            nt.links.new(geo.outputs["Position"], sep.inputs[0])
            div = nt.nodes.new("ShaderNodeMath")
            div.operation = "DIVIDE"
            div.inputs[1].default_value = HEIGHT_SCALE_M
            nt.links.new(sep.outputs["Z"], div.inputs[0])
            nt.links.new(div.outputs[0], comb.inputs[2])
            nt.links.new(comb.outputs[0], em.inputs["Color"])
        elif kind == "ao":
            # R: ambient occlusion, G: clear coat weight
            ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
            ao.inputs["Distance"].default_value = 0.35
            ao.samples = 16
            comb = nt.nodes.new("ShaderNodeCombineColor")
            nt.links.new(ao.outputs["AO"], comb.inputs[0])
            coat = p.inputs.get("Coat Weight")
            comb.inputs[1].default_value = coat.default_value if coat is not None else 0.0
            nt.links.new(comb.outputs[0], em.inputs["Color"])
        # Keep the material's alpha (decal edges, cut-outs)
        a = source(p.inputs["Alpha"])
        mix = nt.nodes.new("ShaderNodeMixShader")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        if a is not None:
            nt.links.new(a, mix.inputs["Fac"])
        else:
            mix.inputs["Fac"].default_value = p.inputs["Alpha"].default_value
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs["Surface"])


def restore():
    for name, o in orig.items():
        mat = bpy.data.materials[name]
        if o["surface"] is not None:
            mat.node_tree.links.new(o["surface"], o["out"].inputs["Surface"])


def render(name: str, samples: int) -> None:
    cyc.samples = samples
    r.filepath = str(OUT / f"{name}.png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", r.filepath, flush=True)


# ---------------------------------------------------------------------------------- lights


def studio():
    """A soft overhead box plus a dim world, for the beauty check."""
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.03, 0.035, 1)
    scene.world = world
    for loc, power, size in (
        ((0, -length_m * 0.45, 6.0), 6000.0, 4.0),
        ((3.5, -1.0, 3.0), 900.0, 2.0),
    ):
        ld = bpy.data.lights.new("key", "AREA")
        ld.energy = power
        ld.size = size
        lo_ = bpy.data.objects.new("key", ld)
        lo_.location = loc
        lo_.rotation_euler = (
            (Vector(loc) - Vector((0, -length_m * 0.45, 0))).to_track_quat("Z", "Y").to_euler()
        )
        scene.collection.objects.link(lo_)


def photographic():
    """AgX tone mapping with a punchier look, like a camera, for the light stage."""
    vs = scene.view_settings
    vs.view_transform = "AgX"
    for look in ("AgX - Punchy", "AgX - Medium High Contrast", "Punchy", "None"):
        try:
            vs.look = look
            return
        except TypeError:
            continue


def light_stage() -> dict:
    """Render the softbox grid and the ambient photo; returns ``stage.json``'s content."""
    stage_dir = OUT / "stage"
    stage_dir.mkdir(exist_ok=True)
    # Frame only the front of the car: from MARGIN_M above the nose to STAGE_LENGTH_M behind
    top, bottom = MARGIN_M, -STAGE_LENGTH_M
    w_m, h_m = img_w_m, top - bottom
    cam_data.ortho_scale = max(w_m, h_m)
    cam.location = (0.0, (top + bottom) / 2, hi.z + 5.0)
    r.resolution_x = int(round(w_m * STAGE_PX_PER_M))
    r.resolution_y = int(round(h_m * STAGE_PX_PER_M))
    r.image_settings.color_depth = "8"
    photographic()
    cyc.use_denoising = True
    for o in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)
    world = bpy.data.worlds.new("stage")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world

    def softbox(name, size, power):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "DISK"
        ld.size = size
        ld.energy = power
        ob = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(ob)  # area lights face -z: straight down
        return ob

    amb = softbox("ambient", 7.0, STAGE_AMBIENT_POWER_W)
    amb.location = (0.0, (top + bottom) / 2, 4.0)
    render("stage/ambient", SAMPLES_STAGE)
    amb.hide_render = True
    box = softbox("softbox", STAGE_SIZE_M, STAGE_POWER_W)
    cells = [(iy, ix) for iy in range(len(STAGE_YS)) for ix in range(len(STAGE_XS))]
    if STAGE_TEST:
        cells = [(2, 2), (4, 1), (6, 3)]
    for iy, ix in cells:
        box.location = (STAGE_XS[ix], MARGIN_M - STAGE_YS[iy], STAGE_LIGHT_Z_M)
        render(f"stage/light_{iy}_{ix}", SAMPLES_STAGE)
    return {
        "px_per_m": STAGE_PX_PER_M,
        "margin_m": MARGIN_M,
        "image_px": [r.resolution_x, r.resolution_y],
        "xs": list(STAGE_XS),
        "ys": list(STAGE_YS),
        "light_z_m": STAGE_LIGHT_Z_M,
        "softbox_m": STAGE_SIZE_M,
        "power_w": STAGE_POWER_W,
        "finish": FINISH if BLACKOUT else None,
        "complete": not STAGE_TEST,
    }


def split_front_wheels():
    """Move the front wheels (tyre, rim and disc faces at the axle nearer the nose) into
    their own object. Returns it, the wheel centres and both axles (world x, y)."""
    import bmesh

    body = meshes[0]
    mats = [s.material.name.split(".")[0] if s.material else "" for s in body.material_slots]
    wheel_slots = {i for i, m in enumerate(mats) if m in WHEEL_MATS}
    bm = bmesh.new()
    bm.from_mesh(body.data)
    mw = body.matrix_world
    faces = [f for f in bm.faces if f.material_index in wheel_slots]
    ys = [(mw @ f.calc_center_median()).y for f in faces]
    mid = (min(ys) + max(ys)) / 2  # front axle: y > mid (the nose is at y = 0)
    front = [f for f, y in zip(faces, ys, strict=True) if y > mid]
    rear = [f for f, y in zip(faces, ys, strict=True) if y <= mid]

    def centres(group):
        out = []
        for side in (-1.0, 1.0):
            pts = [mw @ v.co for f in group for v in f.verts if (mw @ v.co).x * side > 0]
            xs_ = [q.x for q in pts]
            ys_ = [q.y for q in pts]
            out.append(
                {
                    "x": (min(xs_) + max(xs_)) / 2,
                    "y": (min(ys_) + max(ys_)) / 2,
                    "r": max(max(xs_) - min(xs_), max(ys_) - min(ys_)) / 2,
                }
            )
        return out

    front_c, rear_c = centres(front), centres(rear)
    # Everything inside each front wheel's volume turns with it: the tyre and rim found by
    # material above, plus the rim covers, brake ducts and upright (other materials)
    for f in bm.faces:
        f.select = False
    for f in bm.faces:
        p = mw @ f.calc_center_median()
        for c in front_c:
            if (
                abs(p.x - c["x"]) <= WHEEL_HALF_WIDTH_M + WHEEL_GRAB_M
                and abs(p.y - c["y"]) <= c["r"] + WHEEL_GRAB_M
                and p.z <= 2 * c["r"] + WHEEL_GRAB_M
            ):
                f.select = True
                break
    front = [f for f in bm.faces if f.select]
    bm.to_mesh(body.data)
    bm.free()
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    wheels = next(o for o in scene.objects if o.type == "MESH" and o is not body)
    wheels.name = "front_wheels"
    print("FRONT_WHEELS", len(front), "faces", flush=True)
    return body, wheels, front_c, rear_c


def drive_stage() -> dict:
    """The driving stage (``--drive``); returns ``stage.json``'s content."""
    out_dir = OUT / "drive_stage"
    out_dir.mkdir(exist_ok=True)
    body, wheels, front_c, rear_c = split_front_wheels()
    top, bottom = MARGIN_M, -(length_m + MARGIN_M)
    w_m, h_m = img_w_m, top - bottom
    cam_data.ortho_scale = max(w_m, h_m)
    cam.location = (0.0, (top + bottom) / 2, hi.z + 5.0)
    r.resolution_x = int(round(w_m * DRIVE_PX_PER_M))
    r.resolution_y = int(round(h_m * DRIVE_PX_PER_M))
    r.image_settings.color_depth = "8"
    photographic()
    cyc.use_denoising = True
    for o in [o for o in scene.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)
    world = bpy.data.worlds.new("drive")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, (top + bottom) / 2, 0.0))
    ground = bpy.context.active_object
    ground.is_shadow_catcher = True

    def softbox(name, size, power):
        ld = bpy.data.lights.new(name, "AREA")
        ld.shape = "DISK"
        ld.size = size
        ld.energy = power
        ob = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(ob)
        return ob

    amb = softbox("ambient", 7.0, STAGE_AMBIENT_POWER_W)
    amb.location = (0.0, (top + bottom) / 2, 4.0)
    box = softbox("softbox", STAGE_SIZE_M, STAGE_POWER_W)
    cells = [(iy, ix) for iy in range(len(DRIVE_YS)) for ix in range(len(STAGE_XS))]
    if STAGE_TEST:
        cells = [(2, 2), (5, 1), (8, 3)]
    for layer in ("body", "wheel"):
        # Body: the wheels only cast shadows; wheels: the body only casts shadows, no ground
        body.visible_camera = layer == "body"
        wheels.visible_camera = layer == "wheel"
        # The ground shadow comes from the broad ambient light only (a soft contact shadow):
        # under the low softbox it was huge and swung with the cursor. The front wheels cast
        # none, as theirs would stay straight while they steer.
        wheels.visible_shadow = layer == "wheel"
        ground.hide_render = layer == "wheel"
        amb.hide_render, box.hide_render = False, True
        render(f"drive_stage/{layer}_ambient", SAMPLES_STAGE)
        ground.hide_render = True
        amb.hide_render, box.hide_render = True, False
        for iy, ix in cells:
            box.location = (STAGE_XS[ix], MARGIN_M - DRIVE_YS[iy], STAGE_LIGHT_Z_M)
            render(f"drive_stage/{layer}_{iy}_{ix}", SAMPLES_STAGE)

    def to_map(c):
        return {"x": c["x"], "y": MARGIN_M - c["y"], "r": c["r"]}

    return {
        "px_per_m": DRIVE_PX_PER_M,
        "margin_m": MARGIN_M,
        "image_px": [r.resolution_x, r.resolution_y],
        "xs": list(STAGE_XS),
        "ys": list(DRIVE_YS),
        "light_z_m": STAGE_LIGHT_Z_M,
        "softbox_m": STAGE_SIZE_M,
        "power_w": STAGE_POWER_W,
        "finish": FINISH if BLACKOUT else None,
        "wheels": [to_map(c) for c in front_c],
        "front_axle_y": MARGIN_M - sum(c["y"] for c in front_c) / 2,
        "rear_axle_y": MARGIN_M - sum(c["y"] for c in rear_c) / 2,
        "length_m": length_m,
        "complete": not STAGE_TEST,
    }


recolour()
delete_decals()
set_surfaces()
if BLACKOUT:
    blackout()
    apply_finish(FINISH)
if DRIVE:
    info = drive_stage()
    (OUT / "drive_stage" / "stage.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("DRIVE_STAGE", json.dumps(info), flush=True)
    bpy.ops.wm.quit_blender()
if STAGE:
    info = light_stage()
    (OUT / "stage" / "stage.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print("STAGE", json.dumps(info), flush=True)
    bpy.ops.wm.quit_blender()  # a clean exit (SystemExit makes Blender return 255)
studio()
render("beauty", SAMPLES_BEAUTY)
if not PREVIEW:
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    cyc.use_denoising = False  # data passes: no denoiser smearing
    for kind in ("albedo", "normal", "surface", "ao"):
        # Colour is stored display-encoded (sRGB, more precision in the darks); data passes
        # raw: "Standard" would gamma-encode roughness, normals and heights too
        scene.view_settings.view_transform = "Standard" if kind == "albedo" else "Raw"
        set_emission(kind)
        render(kind, SAMPLES_DATA)
        restore()

meta = {
    "px_per_m": PX_PER_M,
    "margin_m": MARGIN_M,
    "height_scale_m": HEIGHT_SCALE_M,
    "image_px": [r.resolution_x, r.resolution_y],
    "width_m": width_m,
    "length_m": length_m,
    "height_m": hi.z - lo.z,
    "rotated_deg": math.degrees(rot.z),
    "model": Path(MODEL).name,
    "decals_removed": list(DECALS),
    "blackout": BLACKOUT,
    "finish": FINISH if BLACKOUT else None,
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("META", json.dumps(meta), flush=True)
