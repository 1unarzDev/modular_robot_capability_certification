"""Blender render of an assembly from catalog geometry (Fig. 1a). Runs inside Blender, headless:

    scripts/heavy.sh timeout 300 blender -b --factory-startup -P scripts/analysis/render_modules.py -- \
        scripts/analysis/assets/teaser_episode.json scripts/analysis/assets/teaser_render.png

Every dimension comes from the JSON written by teaser_episode.py (capxfer/assembly.py:build and the params file):
module centres on the dock pitch, wheel centres at (+-half_wheelbase, +-half_track), 45 deg rollers whose contact
axis is the assembly's r_hat (pattern fl/rr "/", fr/rl "\\" before the module yaw), wheel radius. Housing size
(0.24 x 0.24 x 0.11 m), 0.02 m ground clearance, 0.04 m coupler projection per face and the forward camera bracket
are from modular/reference/physical_context.md (descriptive). Housings are drawn translucent so the wheels inside
the body, including the faulted one, are visible. Cycles CPU, transparent film with a shadow catcher.
"""
import json
import math
import sys

import bpy
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, OUT = argv[0], argv[1]
RES_X, RES_Y = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (1500, 1100)
SAMPLES = int(argv[4]) if len(argv) > 4 else 96
E = json.load(open(SRC))
A = E["parent"]

HOUSING = (0.24, 0.24, 0.11)        # physical_context.md chassis_m
CLEAR = 0.02                        # ground_clearance_m
COUPLER = 0.04                      # dock_projection_per_face_m
FACE_H = 0.06                       # receiver face_height_m
R = E["wheel_radius_m"]
WHEEL_W = 0.034
N_ROLL = E["rollers"]


def hexrgb(h, a=1.0):
    h = h.lstrip("#")
    srgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb]
    return (*lin, a)


def material(name, hex_, rough=0.55, alpha=1.0, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = hexrgb(hex_)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except AttributeError:
            pass
    return m


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
M_HOUSE = material("housing", "#e8e6e1", rough=0.3, alpha=0.14)
M_EDGE = material("edge", "#52514e", rough=0.6)
M_COUP = material("coupler", "#7d7b76", rough=0.5)
M_CAM = material("camera", "#2b2b2a", rough=0.4)
M_HUB = material("hub", "#3a3a38", rough=0.45, metal=0.3)
M_ROLL = material("roller", "#4a4946", rough=0.6)
M_FAULT = material("fault", "#eb6834", rough=0.5)
M_LIDAR = material("lidar", "#2b2b2a", rough=0.3)



def box(name, size, loc, mat, rot_z=0.0, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = size
    o.rotation_euler[2] = rot_z
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    if bevel:
        md = o.modifiers.new("bev", "BEVEL")
        md.width, md.segments = bevel, 2
    o.data.materials.append(mat)
    return o


def module_frame(k):
    th = A["modules"][k][3] * math.pi / 2
    c, s = math.cos(th), math.sin(th)
    return (lambda v: Vector((c * v[0] - s * v[1], s * v[0] + c * v[1])))


def build_module(k):
    mid = A["modules"][k][0]
    cx, cy = A["module_pos"][k]
    rot = module_frame(k)
    th = A["modules"][k][3] * math.pi / 2
    zc = CLEAR + HOUSING[2] / 2
    h = box(f"house{k}", HOUSING, (cx, cy, zc), M_HOUSE, th, bevel=0.006)
    # outline: a thin opaque wireframe of the housing
    bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, cy, zc))
    w = bpy.context.object
    w.scale = HOUSING
    w.rotation_euler[2] = th
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    md = w.modifiers.new("wire", "WIREFRAME")
    md.thickness, md.use_even_offset = 0.004, True
    w.data.materials.append(M_EDGE)
    # one coupler per face (hooks + magnets), centred, FACE_H tall, projecting COUPLER
    for f in range(4):
        n = rot((1, 0)) if f == 0 else rot((0, 1)) if f == 1 else rot((-1, 0)) if f == 2 else rot((0, -1))
        p = Vector((cx, cy)) + n * (HOUSING[0] / 2 + COUPLER / 2)
        ang = math.atan2(n[1], n[0])
        box(f"coup{k}_{f}", (COUPLER, 0.10, FACE_H), (p[0], p[1], CLEAR + 0.025 + FACE_H / 2), M_COUP, ang, 0.003)
    # forward camera bracket on the top front edge (module x = front): shows the module yaw
    p = Vector((cx, cy)) + rot((0.10, 0.0))
    box(f"cam{k}", (0.035, 0.07, 0.028), (p[0], p[1], CLEAR + HOUSING[2] + 0.014), M_CAM, th, 0.003)
    if A["lidar"][k]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=0.04,
                                            location=(cx, cy, CLEAR + HOUSING[2] + 0.02))
        bpy.context.object.data.materials.append(M_LIDAR)
    return h


def build_wheel(i, faulted):
    px, py = A["wheel_pos"][i]
    wv = Vector((*A["w_hat"][i], 0.0)).normalized()          # rolling direction
    ax = Vector((-wv[1], wv[0], 0.0))                         # axle (left of rolling direction)
    rv = Vector((*A["r_hat"][i], 0.0))
    rx, ry = rv.dot(wv), rv.dot(ax)                           # contact-roller axis in (rolling, axle)
    zv = Vector((0, 0, 1))
    c = Vector((px, py, R))
    hub_mat = M_FAULT if faulted else M_HUB
    roll_mat = M_FAULT if faulted else M_ROLL
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.030, depth=WHEEL_W, location=c)
    hub = bpy.context.object
    hub.rotation_euler = ax.to_track_quat("Z", "Y").to_euler()
    hub.data.materials.append(hub_mat)
    rr, rl = 0.0125, 0.040
    for q in range(N_ROLL):
        phi = 2 * math.pi * q / N_ROLL
        radial = -math.cos(phi) * zv + math.sin(phi) * wv      # phi = 0: contact point
        tang = math.cos(phi) * wv + math.sin(phi) * zv
        axis = (rx * tang + ry * ax).normalized()
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, radius=1,
                                             location=c + radial * (R - rr))
        o = bpy.context.object
        o.scale = (rr, rr, rl / 2)
        o.rotation_euler = axis.to_track_quat("Z", "Y").to_euler()
        o.data.materials.append(roll_mat)
        bpy.ops.object.shade_smooth()


fault_wheels = {f["wheel"] for f in E["faults"]}
for k in range(len(A["modules"])):
    build_module(k)
for i in range(len(A["wheel_pos"])):
    build_wheel(i, i in fault_wheels)

# ground: shadow catcher
bpy.ops.mesh.primitive_plane_add(size=6, location=(0, 0, 0))
g = bpy.context.object
g.is_shadow_catcher = True

# lights and world
bpy.ops.object.light_add(type="SUN", location=(0, 0, 3))
sun = bpy.context.object
sun.data.energy = 3.2
sun.data.angle = math.radians(12)
sun.rotation_euler = (math.radians(38), math.radians(-12), math.radians(-35))
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9

# camera: orthographic, from the front-right, high enough to see into the dock gaps
xs = [p[0] for p in A["module_pos"]]
ys = [p[1] for p in A["module_pos"]]
cen = Vector(((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2, 0.06))
az, el = math.radians(-38), math.radians(52)
d = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))
bpy.ops.object.camera_add(location=cen + 4 * d)
cam = bpy.context.object
cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
cam.data.type = "ORTHO"
sc.camera = cam
# fit the ortho scale to the projected footprint (+ margin)
bpy.context.view_layer.update()
inv = cam.matrix_world.inverted()
pts = []
for (x, y) in A["module_pos"]:
    for dx in (-0.17, 0.17):
        for dy in (-0.17, 0.17):
            for z in (0.0, 0.16):
                pts.append(inv @ Vector((x + dx, y + dy, z)))
w = max(p[0] for p in pts) - min(p[0] for p in pts)
hgt = max(p[1] for p in pts) - min(p[1] for p in pts)
mx = (max(p[0] for p in pts) + min(p[0] for p in pts)) / 2
my = (max(p[1] for p in pts) + min(p[1] for p in pts)) / 2
cam.data.ortho_scale = 1.02 * max(w, hgt * RES_X / RES_Y)
cam.data.shift_x = mx / cam.data.ortho_scale
cam.data.shift_y = my / cam.data.ortho_scale

sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = SAMPLES
sc.cycles.use_denoising = True
sc.render.threads_mode = "FIXED"
sc.render.threads = 4
sc.render.film_transparent = True
sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
sc.render.resolution_percentage = 100
sc.view_settings.view_transform = "Standard"
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)

# camera projection of the faulted wheel(s) and module centres -> sidecar JSON for the figure's leaders
from bpy_extras.object_utils import world_to_camera_view  # noqa: E402
proj = {}
for i in sorted(fault_wheels):
    v = world_to_camera_view(sc, cam, Vector((*A["wheel_pos"][i], R)))
    proj[f"wheel{i}"] = [v[0], v[1]]
for k, (x, y) in enumerate(A["module_pos"]):
    v = world_to_camera_view(sc, cam, Vector((x, y, CLEAR + HOUSING[2])))
    proj[f"module{k}"] = [v[0], v[1]]
json.dump(dict(res=[RES_X, RES_Y], proj=proj), open(OUT.rsplit(".", 1)[0] + "_proj.json", "w"), indent=1)
print("wrote", OUT)
