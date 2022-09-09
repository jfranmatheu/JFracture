from collections import defaultdict
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
import platform

from mathutils.bvhtree import BVHTree
from bpy.types import Object, Collection
from mathutils import Vector, Matrix
from mathutils.geometry import points_in_planes
import bmesh
from bmesh.types import BMFace
from bmesh.ops import remove_doubles, convex_hull, dissolve_limit

from jfracture.decorators import timer_it

user_os = platform.system()

if user_os not in {'Windows', 'Linux'}:
    print("Error! Operative System not supported!")
    sys.exit()

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

instance_uid: int = int(sys.argv[-2])  # Un identificador numérico único.
object_names: List[str] = sys.argv[-1].split(',')  # Lista de nombres de objects.

MODULE_DIR = dirname(abspath(__file__))
SETTINGS_PATH = join(MODULE_DIR, 'settings.json')
TMP_DIR = gettempdir()
SRC_PATH = join(TMP_DIR, 'coppybuffer' + str(instance_uid) + '.blend')
DST_PATH = join(TMP_DIR, 'pastebuffer' + str(instance_uid) + '.blend')


FRACTURE_METHOD = {'CUSTOM_CF'}  # {'CYTHON', 'CUSTOM_CF'} # {'CF'}
USE_ONE_BMESH = True


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

# bpy.ops.object.transform_apply(False, location=False, rotation=False, scale=True)


# FRACTURE UTILS.
@timer_it
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


@timer_it
def cy_points_as_bmesh_cells(verts: List[Vector], points: List[Vector]) -> List[Tuple[Vector, List[Vector]]]:
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


@timer_it
def points_as_bmesh_cells(src_object: Object, points: List[Vector]) -> List[Tuple[Vector, List[Vector]]]:
    cells_data = []

    points_sorted_current = [*points]
    plane_indices = []
    vertices = []

    margin_cells: float = settings['margin']
    margin_bounds: float = 0.01

    # Get planes for convex hull via bounding box.
    xa, ya, za = zip(*[Vector(tuple(v)) @ src_object.matrix_world for v in src_object.bound_box])

    xmin, xmax = min(xa) - margin_bounds, max(xa) + margin_bounds
    ymin, ymax = min(ya) - margin_bounds, max(ya) + margin_bounds
    zmin, zmax = min(za) - margin_bounds, max(za) + margin_bounds

    for point_cell_current in points:
        planes = [
            Vector((+1.0, 0.0, 0.0, -xmax)),
            Vector((-1.0, 0.0, 0.0, +xmin)),
            Vector((0.0, +1.0, 0.0, -ymax)),
            Vector((0.0, -1.0, 0.0, +ymin)),
            Vector((0.0, 0.0, +1.0, -zmax)),
            Vector((0.0, 0.0, -1.0, +zmin)),
        ]
        for j in range(6):
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
            plane[3] = (-nlength / 2.0) + margin_cells
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


@timer_it
def cell_fracture_objects(context,
                          collection: Collection,
                          src_object: Object,
                          ori_loc: Vector) -> List[Object]:
    depsgraph = context.evaluated_depsgraph_get()
    src_mesh = src_object.data
    src_material_slots = src_object.material_slots
    inner_mat_slot = src_material_slots[-1]
    outer_mat_slot = src_material_slots[-2]

    cell_name = src_object.name + "_cell_"

    # Get points.
    points = points_from_object(depsgraph, src_object, settings['source'])
    if not points:
        return None

    # Clamp points.
    if settings['source_limit'] != 0 and settings['source_limit'] < len(points):
        random.shuffle(points)
        points = points[:settings['source_limit']]

    # Avoid duplicated points.
    to_tuple = Vector.to_tuple
    points = list({to_tuple(p, 4): p for p in points}.values())  # list({to_tuple(p, 4): p for p in points}.values())

    # Get cell data.
    mesh = src_object.data
    matrix = src_object.matrix_world.copy()
    verts = [matrix @ v.co for v in mesh.vertices]
    if 'CYTHON' in FRACTURE_METHOD:
        cells = cy_points_as_bmesh_cells(verts, points)
    else:
        cells = points_as_bmesh_cells(src_object, points)

    # Get planes for convex hull via bounding box.
    margin_bounds: float = 0.05
    xa, ya, za = zip(*[Vector(tuple(v)) @ src_object.matrix_world for v in src_object.bound_box])

    xmin, xmax = min(xa) - margin_bounds, max(xa) + margin_bounds
    ymin, ymax = min(ya) - margin_bounds, max(ya) + margin_bounds
    zmin, zmax = min(za) - margin_bounds, max(za) + margin_bounds
    def is_outbounds(co): return xmin <= co.x <= xmax and ymin <= co.y <= ymax and zmin <= co.z <= zmax

    # Create the convex hulls.
    new_bmesh = bmesh.new

    src_bmesh = new_bmesh()
    copy_src_mesh = src_mesh.copy()
    src_bmesh.from_mesh(copy_src_mesh)
    # src_bmesh.transform(src_object.matrix_world)
    src_BVHtree = BVHTree.FromBMesh(src_bmesh)

    cell_objects = []
    boundary_cells = []
    new_mesh = bpy.data.meshes.new
    new_object = bpy.data.objects.new
    cell_idx: int = 1
    for center_point, cell_points in cells:
        # New bmesh with the calculated cell points.
        bm = new_bmesh()
        bm_vert_add = bm.verts.new
        {bm_vert_add(co+random_vector()) for co in cell_points}

        # Remove possible double vertices.
        remove_doubles(bm, verts=bm.verts, dist=0.005)

        # Create convex hull from added vertices.
        hull = convex_hull(bm, input=bm.verts, use_existing_faces=False)
        #print("* HULL-", cell_idx)
        #if hull['geom'] != []: print("\t- Geo:", hull['geom'])
        #if hull['geom_interior'] != []: print("\t- Interior:", hull['geom_interior'])
        #if hull['geom_unused'] != []: print("\t- Unused:", hull['geom_unused'])
        #if hull['geom_holes'] != []: print("\t- Holes:", hull['geom_holes'])
        if hull["geom_unused"]:  # hull["geom_holes"]
            bmesh.ops.delete(bm, geom=hull["geom_unused"], context='VERTS')  # + hull["geom_interior"]

        dissolve_limit(bm, verts=bm.verts, edges=bm.edges, angle_limit=0.0025)

        if len(bm.faces) < 3:
            bm.free()
            del bm
            continue

        '''
        fm = bm.faces.layers.face_map.verify()

        for face in bm.faces:
            face_idx = face.index
            map_idx = face[fm]
        '''

        # Check if mesh intersects with the original mesh.
        # TODO: if src_object is cuboid, check bounds overlapping instead.
        # offset cell to be at its real position temporarily.
        # offset = center_point + src_object.location
        bm.transform(Matrix.Translation(center_point))  # <<  Add .name on both lines.
        cell_BVHtree = BVHTree.FromBMesh(bm)
        inter = src_BVHtree.overlap(cell_BVHtree)
        cell_is_boundary: bool = inter != []
        del cell_BVHtree
        # return it back to its original position.
        bm.transform(Matrix.Translation(-center_point))

        # Asign materials to faces.
        if cell_is_boundary:
            for bm_face in bm.faces:
                if all([is_outbounds(v.co) for v in bm_face.verts]):
                    #cell_has_outer = True
                    bm_face.tag = True
                    bm_face.select = True
                    bm_face.material_index = 1  # Outer
                else:
                    bm_face.select = False

        # Create NEW MESH from bmesh.
        cell_mesh = new_mesh(name=cell_name+str(cell_idx))
        bm.to_mesh(cell_mesh)
        bm.free()
        del bm

        # Add materials to new mesh.
        cell_mesh.materials.append(inner_mat_slot.material)  # Inner.
        if cell_is_boundary:
            cell_mesh.materials.append(outer_mat_slot.material)  # Outer.

        # for lay_attr in ("vertex_colors", "uv_layers"):
        #    lay_src = getattr(src_mesh, lay_attr)
        #    lay_dst = getattr(cell_mesh, lay_attr)
        #    for key in lay_src.keys():
        #        lay_dst.new(name=key)

        # Create NEW OBJECT.
        cell_ob = new_object(name=cell_name, object_data=cell_mesh)
        collection.objects.link(cell_ob)
        cell_ob.location = ori_loc + center_point
        cell_ob.select_set(True)
        cell_ob['boundary'] = cell_is_boundary

        # Add material slots to new object.
        slot_inner = cell_ob.material_slots[-1]
        slot_inner.link = inner_mat_slot.link
        slot_inner.material = inner_mat_slot.material
        if cell_is_boundary:
            slot_outer = cell_ob.material_slots[-2]
            slot_outer.link = outer_mat_slot.link
            slot_outer.material = outer_mat_slot.material

        cell_objects.append(cell_ob)
        if cell_is_boundary:
            boundary_cells.append(cell_ob)
        cell_idx += 1

    src_bmesh.free()
    del src_bmesh
    del src_BVHtree
    del cells

    # print(boundary_cells)
    return cell_objects, boundary_cells


@timer_it
def cell_fracture_boolean(context,
                          collection: Collection,
                          src_object: Object,
                          cell_objects: List[Object]) -> List[Object]:
    if not cell_objects:
        print("No cells to apply booleans!?")
        return

    @timer_it
    def add_bool_mod(cell_ob: Object):
        # TODO: add boolean ONLY to boundary cells.
        mod = cell_ob.modifiers.new(name="Boolean", type='BOOLEAN')
        # mod.solver = 'FAST' # SHIT BOOLEANS.
        mod.object = src_object
        mod.operation = 'INTERSECT'

    bpy.ops.object.select_all(False, action='DESELECT')
    for cell_ob in cell_objects:
        cell_ob.select_set(True)

    first_cell_ob = cell_objects[0]
    add_bool_mod(first_cell_ob)
    context.view_layer.objects.active = first_cell_ob
    bpy.ops.object.make_links_data(False, type='MODIFIERS')

    if settings['apply_boolean']:
        # TODO: apply boolean to boundary cells.
        pass

    return cell_objects


@timer_it
def fix_cells_pivot_point(cells: List[Object]) -> None:
    with context.temp_override(selected_editable_objects=cells):
        bpy.ops.object.origin_set(
            #{"selected_editable_objects": cells},
            False,
            type='ORIGIN_GEOMETRY',
            center='MEDIAN'
        )


@timer_it
def find_cell_neighbours(context, cells: List[Object]) -> None:
    # context.view_layer.update()
    depsgraph = context.evaluated_depsgraph_get()
    ray_cast = context.scene.ray_cast
    ray_cast_distance: float = max(0.05, settings['margin'] * 2.0)  # settings['margin'] * 10.0

    cell_neighbours = {cell_ob: set() for cell_ob in cells}
    for cell_ob in cells:
        mw = cell_ob.matrix_world
        cell_ob.hide_set(True)
        #bm = bmesh.new()
        # bm.from_mesh(cell_ob.data)
        # bm.transform(cell_ob.matrix_world)
        # bm.normal_update()

        #bm_faces: List[BMFace] = bm.faces
        for poly in cell_ob.data.polygons:
            # if bm_face.tag:
            # boundary face.
            #    continue
            # bm_face.hide_set(True)

            origin: Vector = mw @ poly.center  # bm_face.calc_center_median()
            normal: Vector = poly.normal  # bm_face.normal.normalized()
            #normal_len: float = normal.length
            #print(origin, normal, normal_len)
            hit, loc, norm, idx, hit_ob, mat = ray_cast(
                depsgraph,
                origin-normal*ray_cast_distance,
                normal,
                distance=ray_cast_distance*2.0)
            if not hit:
                continue
            if hit_ob == cell_ob:
                print("NOPE!")
                continue
            if hit_ob not in cell_neighbours:
                continue
            if hit_ob in cell_neighbours[cell_ob]:
                continue
            cell_neighbours[cell_ob].add(hit_ob)
            cell_neighbours[hit_ob].add(cell_ob)

            # bm_face.hide_set(False)

        cell_ob.hide_set(False)

        # bm.free()
        #del bm

    for cell, neighbours in cell_neighbours.items():
        cell['neighbours'] = ','.join([ob.name for ob in neighbours])

    del cell_neighbours

    '''
    for cell_ob_now in cells:
        #create bmesh objects
        bm1 = bmesh.new()
        bm1.from_mesh(cell_ob_now.data)   #<<  Add .name on both lines.
        bm1.transform(cell_ob_now.matrix_world)   #<<  Add .name on both lines.

        for cell_ob_next in cells:
            if cell_ob_now == cell_ob_next:
                continue


            bm2 = bmesh.new()

            #fill bmesh data from objects

            bm2.from_mesh(obj_next.data)  #<<  Add .name on both lines.

            #fixed it here:
            bm1.transform(cell_ob_now.matrix_world)   #<<  Add .name on both lines.
            bm2.transform(obj_next.matrix_world)  #<<  Add .name on both lines.

            #make BVH tree from BMesh of objects
            obj_now_BVHtree = BVHTree.FromBMesh(bm1)
            obj_next_BVHtree = BVHTree.FromBMesh(bm2)

            #get intersecting pairs
            inter = obj_now_BVHtree.overlap(obj_next_BVHtree)

            #if list is empty, no objects are touching
            if inter != []:
                print(obj_now.name + " and " + obj_next.name + " are touching!")   # <<  Add .name on both lines.
            else:
                print(obj_now.name + " and " + obj_next.name + " NOT touching!")  #<<  Add .name on both lines.
    '''


@timer_it
def fracture(context, to_fracture_ob: Object, collection: Collection) -> None:
    # Move to 3d viewport origin.
    loc_offset: Vector = to_fracture_ob.location.copy()
    to_fracture_ob.location = 0, 0, 0

    # Create fractures.
    cells, boundary_cells = cell_fracture_objects(context, collection, to_fracture_ob, loc_offset)
    to_fracture_ob.location = loc_offset
    if not cells:
        return

    # Add boolean mods to boundary cells.
    cell_fracture_boolean(context, collection, to_fracture_ob, boundary_cells)

    to_fracture_ob.hide_set(True)

    # Center pivot points.
    # settings['apply_boolean'] and
    if settings['use_recenter']:
        fix_cells_pivot_point(cells)

    find_cell_neighbours(context, cells)


@timer_it
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


@timer_it
def setup_source_object_materials(context, ob: Object) -> None:
    # Ensure object is active and selected.
    context.view_layer.objects.active = ob

    # if len(ob.material_slots) == 0:
    bpy.ops.object.material_slot_add(False)
    bpy.ops.object.material_slot_add(False)
    ob.material_slots[0].material = bpy.data.materials.new('Outer__' + ob.name)
    ob.material_slots[1].material = bpy.data.materials.new('Inner__' + ob.name)


@timer_it
def write_output(collections: Set[Collection]) -> None:
    bpy.data.libraries.write(DST_PATH, collections)


# LOOP.
with context.temp_override(**override_ctx):
    print("[Client-%i] Started." % instance_uid)

    output_collections: Set[Collection] = set()
    for ob in to_export_objects:
        print("[Client-%i] Fracturing Object... %s" % (instance_uid, ob.name))
        # Create a fracture collection and ensure is active.
        collection = bpy.data.collections.new(ob.name)
        context.scene.collection.children.link(collection)

        # Add material.
        setup_source_object_materials(context, ob)

        # Do fracture.
        #print(ob.name, ob.matrix_world)
        if 'CUSTOM_CF' in FRACTURE_METHOD:
            fracture(context, ob, collection)
        else:
            builtin_fracture(ob, collection)

        context.scene.collection.children.unlink(collection)
        context.view_layer.update()

        output_collections.add(collection)

    write_output(output_collections)

    # SEND SIGNAL OF FINSIHED.
    print("[Client-%i] Done." % instance_uid)
    sys.exit(0)
