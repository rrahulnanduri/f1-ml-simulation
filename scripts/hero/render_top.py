"""Render a car model from directly above into the maps the menu hero lights (run in Blender).

    blender -b --factory-startup -P scripts/hero/render_top.py -- model.glb out_dir \
        [--preview] [--blackout]

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
metal detail survive as shades of black, but no colour does).

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
BLACKOUT_LEVELS = {"tyre_tread": 0.18, "tyre_sides": 0.18}
"""Greyscale multiplier per material with ``--blackout`` (default ``BLACKOUT_DEFAULT``)."""
BLACKOUT_DEFAULT = 0.22
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


recolour()
delete_decals()
set_surfaces()
if BLACKOUT:
    blackout()
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
}
(OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
print("META", json.dumps(meta), flush=True)
