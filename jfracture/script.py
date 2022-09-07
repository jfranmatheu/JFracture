import json
from tempfile import gettempdir
from typing import List, Set, Tuple
import bpy
from bpy import context
import sys
from os.path import join, dirname, abspath
import random
from math import sqrt
import numpy as np

from bpy.types import Object, Collection
from mathutils import Vector
from mathutils.geometry import points_in_planes
import bmesh
from bmesh.ops import remove_doubles, convex_hull, dissolve_limit

'''
sys.argv ->
[
    'c:\\{blender_path}\\blender.exe',
    'C:\\Users\\{user}\\AppData\\Roaming\\Blender Foundation\\Blender\\3.2\\scripts\\addons\\jfracture\\empty.blend',
    '--background',
    '--python',
    'C:\\Users\\{user}\\AppData\\Roaming\\Blender Foundation\\Blender\\3.2\\scripts\\addons\\jfracture\\script.py',
    '0',
    'Cube,Cube.001'
]
'''
# Python.exe --- sys.argv[0]
# print("Args:", sys.argv)
instance_uid: int = int(sys.argv[-2])  # Un identificador numérico único.
object_names: List[str] = sys.argv[-1].split(',')  # Lista de nombres de objects.

MODULE_DIR = dirname(abspath(__file__))
SETTINGS_PATH = join(MODULE_DIR, 'settings.json')
TMP_DIR = gettempdir()
SRC_PATH = join(TMP_DIR, 'coppybuffer' + str(instance_uid) + '.blend')
DST_PATH = join(TMP_DIR, 'pastebuffer' + str(instance_uid) + '.blend')


# OVERRIDE CONTEXT.
def get_view3d_ctx():
    window = context.window_manager.windows[0]
    for area in window.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type != 'WINDOW':
                continue
            return {
                'window': window,
                'area': area,
                'region': region
            }
    return None


override_ctx = get_view3d_ctx()
if override_ctx is None:
    raise Exception()


# LOAD SETTINGS.
settings = {
    'random_seed': True,
    'source': {'PARTICLE_OWN'},
    'source_limit': 1000,
    'noise': 0.05,
    'scale': (1.0, 1.0, 1.0),
    'margin': 0.00001,
    'use_recenter': True,
    'use_interior_vgroup': True,
    'use_interior_hide': False,
    'use_sharp_edges': False,
    'use_sharp_edges_apply': False,
    'apply_boolean': False,
}
with open(SETTINGS_PATH, 'r') as json_file:
    settings.update(**json.loads(json_file.read()))


# LOAD OBJECTS TO FRACTURE FROM .BLEND...
with bpy.data.libraries.load(SRC_PATH) as (data_from, data_to):
    data_to.objects = data_from.objects
to_export_objects: List[Object] = []

for ob_name in object_names:
    ob = bpy.data.objects[ob_name]
    to_export_objects.append(ob)
    context.scene.collection.objects.link(ob)
    ob.select_set(False)
    # if settings['random_seed']:
    #    for ps in ob.particle_systems:
    #        ps.seed += random.randint(0, 9999)


# FRACTURE UTILS.
def points_from_object(depsgraph, src_object: Object, source: Set[str]):
    points = []

    def points_from_verts(ob: Object):
        matrix = ob.matrix_world.copy()
        points.extend([matrix @ v.co for v in ob.data.vertices])

    def points_from_particles(ob: Object):
        ob_eval = ob.evaluated_get(depsgraph)
        points.extend([p.location.copy()
                       for psys in ob_eval.particle_systems
                       for p in psys.particles])
        if not points:
            points_from_verts(src_object)

    # geom own
    if 'VERT_OWN' in source:
        points_from_verts(src_object)

    # geom children
    if 'VERT_CHILD' in source:
        for ob_child in src_object.children:
            points_from_verts(ob_child)

    # geom particles
    if 'PARTICLE_OWN' in source:
        points_from_particles(src_object)

    if 'PARTICLE_CHILD' in source:
        for ob_child in src_object.children:
            points_from_particles(ob_child)

    return points


def cy_points_as_bmesh_cells(verts: List[Vector], points: List[Vector]) -> List[Tuple[Vector, List[Vector]]]:
    #cells_data = []

    #points_sorted_current = [*points]
    #plane_indices = []
    #vertices = []

    margin: float = settings['margin']

    # Get planes for convex hull.
    xa, ya, za = zip(*[v for v in verts])

    xmin, xmax = min(xa) - margin, max(xa) + margin
    ymin, ymax = min(ya) - margin, max(ya) + margin
    zmin, zmax = min(za) - margin, max(za) + margin
    convexPlanes = np.array([
        [+1.0, 0.0, 0.0, -xmax],
        [-1.0, 0.0, 0.0, +xmin],
        [0.0, +1.0, 0.0, -ymax],
        [0.0, -1.0, 0.0, +ymin],
        [0.0, 0.0, +1.0, -zmax],
        [0.0, 0.0, -1.0, +zmin],
    ], dtype=np.float32)

    from jfracture.cy import jfracture_cy
    cells_data = jfracture_cy.get_cells(
        convexPlanes,  # .reshape((6, 4))
        np.array(points, dtype=np.float32),
        margin)

    return cells_data


def points_as_bmesh_cells(verts: List[Vector], points: List[Vector]) -> List[Tuple[Vector, List[Vector]]]:
    cells_data = []

    points_sorted_current = [*points]
    plane_indices = []
    vertices = []

    margin = settings['margin']

    # Get planes for convex hull.
    xa, ya, za = zip(*[v for v in verts])

    xmin, xmax = min(xa) - margin, max(xa) + margin
    ymin, ymax = min(ya) - margin, max(ya) + margin
    zmin, zmax = min(za) - margin, max(za) + margin
    convexPlanes = [
        Vector((+1.0, 0.0, 0.0, -xmax)),
        Vector((-1.0, 0.0, 0.0, +xmin)),
        Vector((0.0, +1.0, 0.0, -ymax)),
        Vector((0.0, -1.0, 0.0, +ymin)),
        Vector((0.0, 0.0, +1.0, -zmax)),
        Vector((0.0, 0.0, -1.0, +zmin)),
    ]

    for point_cell_current in points:
        planes = [None] * 6  # len(convexPlanes)
        for j in range(6):  # len(convexPlanes)
            planes[j] = convexPlanes[j].copy()
            planes[j][3] += planes[j].xyz.dot(point_cell_current)

        points_sorted_current.sort(key=lambda p: (p - point_cell_current).length_squared)

        distance_max = 10000000000.0  # a big value!

        for j in range(1, len(points)):
            normal = points_sorted_current[j] - point_cell_current
            nlength = normal.length

            if nlength > distance_max:
                break

            plane = normal.normalized()
            plane.resize_4d()
            plane[3] = (-nlength / 2.0) + margin
            planes.append(plane)

            vertices[:], plane_indices[:] = points_in_planes(planes)
            if len(vertices) == 0:
                break

            if len(plane_indices) != len(planes):
                planes[:] = [planes[k] for k in plane_indices]

            distance_max = sqrt(max(v.length_squared for v in vertices)) * 2.0  # sqrt -> make real length

        if len(vertices) < 4:
            continue

        cells_data.append((point_cell_current, [*vertices]))
        del vertices[:]

    return cells_data


def random_vector() -> Vector:
    return Vector((
        (random.random() - 0.5) * 0.001,
        (random.random() - 0.5) * 0.001,
        (random.random() - 0.5) * 0.001
    ))


inner_material_index = None

# FRACTURE PROCESS.


def add_basic_material(mat_name: str, obj: Object) -> None:
    if mat_name not in bpy.data.materials:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
    else:
        mat = bpy.data.materials.get(mat_name)

    if 'Inner_mat' in mat_name:
        mat[mat_name] = True

    if mat_name not in obj.data.materials:
        obj.data.materials.append(mat)


def cell_fracture_objects(context, collection: Collection, src_object: Object, clean=True) -> List[Object]:
    depsgraph = context.evaluated_depsgraph_get()
    src_mesh = src_object.data

    global inner_material_index
    mat_inner_name = 'RBDLab_Inner_mat'
    view_layer = context.view_layer

    # Get points.
    points = points_from_object(depsgraph, src_object, settings['source'])

    if not points:
        print("No points found!")
        return []

    # Clamp points.
    if settings['source_limit'] != 0 and settings['source_limit'] < len(points):
        random.shuffle(points)
        points = points[:settings['source_limit']]

    # Avoid duplicated points.
    to_tuple = Vector.to_tuple
    points = list({to_tuple(p, 4): p for p in points}.values())
    del to_tuple

    # boundbox approx of overall scale
    # if settings['noise'] > 0.0:
    #     matrix = src_object.matrix_world.copy()
    #     bb_world = [matrix @ Vector(v) for v in src_object.bound_box]
    #     scalar = settings['noise'] * ((bb_world[0] - bb_world[6]).length / 2.0)
    #     from mathutils.noise import random_unit_vector
    #     points[:] = [p + (random_unit_vector() * (scalar * random())) for p in points]

    # Get cell data.
    mesh = src_object.data
    matrix = src_object.matrix_world.copy()
    verts = [matrix @ v.co for v in mesh.vertices]
    cells = points_as_bmesh_cells(verts, points)

    cell_name = src_object.name + "_cell_"

    # Create the convex hulls.
    cell_objects = []
    new_mesh = bpy.data.meshes.new
    new_object = bpy.data.objects.new
    new_bmesh = bmesh.new
    i: int = 1

    # RBDLab Materials
    mat_outer_name = 'RBDLab_Outer_mat'
    mat_inner_name = 'RBDLab_Inner_mat'

    if len(src_object.material_slots) == 0:
        add_basic_material(mat_outer_name, src_object)
        add_basic_material(mat_inner_name, src_object)
    if len(src_object.material_slots) >= 1:
        add_basic_material(mat_inner_name, src_object)

    for center_point, cell_points in cells:
        # New bmesh with the calculated cell points.
        bm = new_bmesh()
        bm_vert_add = bm.verts.new

        # Small noise
        {bm_vert_add(co+random_vector()) for co in cell_points}

        # Remove possible double vertices.
        remove_doubles(bm, verts=bm.verts, dist=0.005)

        # Create convex hull from added vertices.
        convex_hull(bm, input=bm.verts)

        if len(bm.faces) < 3:
            bm.free()
            del bm
            continue

        if clean:
            bm.normal_update()
            dissolve_limit(bm, verts=bm.verts, angle_limit=0.001)

        if mat_inner_name in src_object.data.materials:
            for bm_face in bm.faces:
                if not inner_material_index:
                    inner_material_index = src_object.material_slots.find(mat_inner_name)
                bm_face.material_index = inner_material_index

        # Asign materials to faces.
        # for bm_face in bm.faces:
        #    bm_face.material_index = 0

        # Create NEW MESH from bmesh.
        mesh_dst = new_mesh(name=cell_name+str(i))
        bm.to_mesh(mesh_dst)
        mesh_dst.update()
        bm.clear()
        bm.free()
        del bm

        # Add materials to new mesh.
        for mat in src_mesh.materials:
            mesh_dst.materials.append(mat)

        # esto tira esto:
        # mesh_ensure_tessellation_customdata: warning! Tessellation uvs or vcol data got out of sync, had to reset!
        # CD_MTFACE: 0 != CD_MLOOPUV: 1 || CD_MCOL: 0 != CD_PROP_BYTE_COLOR: 0
        for lay_attr in ("vertex_colors", "uv_layers"):
            lay_src = getattr(mesh, lay_attr)
            lay_dst = getattr(mesh_dst, lay_attr)
            for key in lay_src.keys():
                lay_dst.new(name=key)

        # Create NEW OBJECT.
        cell_ob = new_object(name=cell_name, object_data=mesh_dst)
        collection.objects.link(cell_ob)
        cell_ob.location = center_point
        cell_ob.select_set(True)

        # Add material slots to new object.
        for i in range(len(mesh_dst.materials)):
            slot_src = src_object.material_slots[i]
            slot_dst = cell_ob.material_slots[i]

            slot_dst.link = slot_src.link
            slot_dst.material = slot_src.material

        cell_objects.append(cell_ob)
        i += 1

    view_layer.update()
    del cells
    return cell_objects


def cell_fracture_boolean(
        context, collection: Collection, src_object: Object, cell_objects: List[Object]) -> List[Object]:
    def add_bool_mod(cell_ob: Object):
        # TODO: add boolean ONLY to boundary cells.
        mod = cell_ob.modifiers.new(name="Boolean", type='BOOLEAN')
        # mod.solver = 'FAST' # SHIT BOOLEANS.
        mod.object = src_object
        mod.operation = 'INTERSECT'

    if cell_objects:
        first_cell_ob = cell_objects[0]
        add_bool_mod(first_cell_ob)
        context.view_layer.objects.active = first_cell_ob
        bpy.ops.object.make_links_data(False, type='MODIFIERS')

        if settings['apply_boolean']:
            # TODO: apply boolean to boundary cells.
            pass

        return cell_objects
    else:
        print('[cell_fracture_boolean]: No Recived Objects!!')
        return


def cell_fracture_interior_handle(cell_objects: List[Object]) -> None:

    global inner_material_index
    mat_inner_name = 'RBDLab_Inner_mat'

    for cell_ob in cell_objects:
        mesh = cell_ob.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        # if settings['use_interior_vgroup']:
        #     for bm_vert in bm.verts:
        #         bm_vert.tag = True
        #     for bm_face in bm.faces:
        #         if not bm_face.hide:
        #             for bm_vert in bm_face.verts:
        #                 bm_vert.tag = False

        #     # now add all vgroups
        #     defvert_lay = bm.verts.layers.deform.verify()
        #     for bm_vert in bm.verts:
        #         if bm_vert.tag:
        #             bm_vert[defvert_lay][0] = 1.0

        #     # add a vgroup
        #     cell_ob.vertex_groups.new(name="Interior")

        # if settings['use_sharp_edges']:
        #     for bm_edge in bm.edges:
        #         if len({bm_face.hide for bm_face in bm_edge.link_faces}) == 2:
        #             bm_edge.smooth = False

        #     if settings['use_sharp_edges_apply']:
        #         edges = [edge for edge in bm.edges if edge.smooth is False]
        #         if edges:
        #             bm.normal_update()
        #             bmesh.ops.split_edges(bm, edges=edges)

        # face_maps:

        fm_lay = bm.faces.layers.face_map.verify()
        fm = cell_ob.face_maps.new(name="Interior")
        # vertex_group_data = []

        for bm_face in bm.faces:
            bm_face.hide = False

            if not inner_material_index:
                inner_material_index = cell_ob.material_slots.find(mat_inner_name)

            if bm_face.material_index == inner_material_index:
                bm_face[fm_lay] = fm.index

        bm.to_mesh(mesh)
        bm.free()
        del bm


def fracture(to_fracture_ob: Object, collection: Collection) -> None:
    objects = cell_fracture_objects(context, collection, to_fracture_ob)
    objects = cell_fracture_boolean(context, collection, to_fracture_ob, objects)

    # Must apply after boolean.
    if settings['apply_boolean'] and settings['use_recenter']:
        bpy.ops.object.origin_set(
            False,
            {"selected_editable_objects": objects},
            type='ORIGIN_GEOMETRY',
            center='MEDIAN',
        )

    if settings['apply_boolean'] and (settings['use_interior_vgroup'] or settings['use_sharp_edges']):
        cell_fracture_interior_handle(objects)


def builtin_fracture(to_fracture_ob: Object, collection: Collection):
    import addon_utils
    addon_utils.enable('object_fracture_cell')

    # Fracture.
    bpy.ops.object.add_fracture_cell_objects(
        False,
        source=settings['source'],
        source_limit=settings['source_limit'],
        source_noise=settings['noise'],
        cell_scale=settings['scale'],
        recursion=0,
        recursion_source_limit=8,
        recursion_clamp=250,
        recursion_chance=0.25,
        recursion_chance_select='SIZE_MIN',
        # use_sharp_edges_apply=use_sharp_edges_apply,
        margin=settings['margin'],
        material_index=len(to_fracture_ob.material_slots)-1,
        # use_interior_vgroup=1, # al usar debug no hace el split ni los grupos
        # use_debug_bool=inner_detail,
        use_debug_redraw=False,
        use_debug_bool=True,
        collection_name=collection.name
    )


# LOOP.
with context.temp_override(**override_ctx):
    print("[Client-%i] Started." % instance_uid)

    output_collections: Set[Collection] = set()
    for ob in to_export_objects:
        print("[Client-%i] Fracturing Object... %s" % (instance_uid, ob.name))
        # Ensure object is active and selected.
        context.view_layer.objects.active = ob
        # ob.select_set(True)

        # Create a fracture collection and ensure is active.
        collection = bpy.data.collections.new(ob.name)
        context.scene.collection.children.link(collection)

        # Do fracture.
        fracture(ob, collection)

        # Deselect
        # ob.select_set(False)

        # Add original object to collection (with a diff name).
        # ob.name = "__FractureSource__" + ob.name
        # collection.objects.link(ob)

        context.scene.collection.children.unlink(collection)

        output_collections.add(collection)

    bpy.data.libraries.write(DST_PATH, output_collections)

    # SEND SIGNAL OF FINSIHED.
    print("[Client-%i] Done." % instance_uid)
