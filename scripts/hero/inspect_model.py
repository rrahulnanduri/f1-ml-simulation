"""Report what a downloaded car model contains (run inside Blender).

    blender -b --factory-startup -P scripts/hero/inspect_model.py -- model.glb report.json

Imports the model into an empty scene and writes a JSON report: overall bounds and size,
each mesh object (vertices, bounds, materials) and each material (its Principled BSDF
inputs and the image textures feeding them). Changes nothing on disk except the report.
"""

import json
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1 :]
model, report_path = argv[0], argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=model)


def world_bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    return lo, hi


meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
lo, hi = world_bounds(meshes)
report = {
    "bounds_min": lo,
    "bounds_max": hi,
    "size": [b - a for a, b in zip(lo, hi, strict=True)],
    "n_mesh_objects": len(meshes),
    "n_vertices": sum(len(o.data.vertices) for o in meshes),
    "objects": [],
    "materials": {},
}
for o in meshes:
    olo, ohi = world_bounds([o])
    report["objects"].append(
        {
            "name": o.name,
            "verts": len(o.data.vertices),
            "min": [round(v, 3) for v in olo],
            "max": [round(v, 3) for v in ohi],
            "materials": [s.material.name for s in o.material_slots if s.material],
        }
    )


def images_into(node, seen=None):
    """Image textures upstream of a node input."""
    seen = seen or set()
    out = []
    for inp in node.inputs:
        for link in inp.links:
            n = link.from_node
            if n in seen:
                continue
            seen.add(n)
            if n.type == "TEX_IMAGE" and n.image:
                out.append(n.image.name)
            out += images_into(n, seen)
    return out


for mat in bpy.data.materials:
    info = {"use_nodes": mat.use_nodes, "blend": getattr(mat, "blend_method", "")}
    if mat.use_nodes:
        for n in mat.node_tree.nodes:
            if n.type == "BSDF_PRINCIPLED":
                for name in (
                    "Base Color",
                    "Metallic",
                    "Roughness",
                    "Normal",
                    "Alpha",
                    "Emission Color",
                ):
                    inp = n.inputs.get(name)
                    if inp is None:
                        continue
                    if inp.is_linked:
                        imgs = []
                        for link in inp.links:
                            fn = link.from_node
                            if fn.type == "TEX_IMAGE" and fn.image:
                                imgs.append(fn.image.name)
                            imgs += images_into(fn)
                        info[name] = {"textures": sorted(set(imgs))}
                    else:
                        v = inp.default_value
                        info[name] = (
                            [round(x, 3) for x in v] if hasattr(v, "__len__") else round(v, 3)
                        )
    report["materials"][mat.name] = info

report["images"] = {im.name: list(im.size) for im in bpy.data.images if im.size[0] > 0}
with open(report_path, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1)
print("REPORT_OK", report_path)
