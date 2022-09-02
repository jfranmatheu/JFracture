import json
import struct
from tempfile import gettempdir
from time import sleep
from typing import List, Set, Union
import bpy
from bpy import context
import sys
import socket
from os.path import join, dirname, abspath


from bpy.types import Object, Collection
import addon_utils
addon_utils.enable('object_fracture_cell')

'''
sys.argv ->
[
    'c:\\{blender_path}\\blender.exe',
    'C:\\Users\\{user}\\AppData\\Roaming\\Blender Foundation\\Blender\\3.2\\scripts\\addons\\jfracture\\empty.blend',
    '--background',
    '--python',
    'C:\\Users\\{user}\\AppData\\Roaming\\Blender Foundation\\Blender\\3.2\\scripts\\addons\\jfracture\\script.py',
    '64387',
    '0',
    'Cube,Cube.001'
]
'''
# Python.exe --- sys.argv[0]
# print("Args:", sys.argv)
port: int = int(sys.argv[-3]) # Puerto de comunicación con el server.
instance_uid: int = int(sys.argv[-2]) # Un identificador numérico único.
object_names: List[str] = sys.argv[-1].split(',') # Lista de nombres de objects.

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
    'source': {'PARTICLE_OWN'},
    'source_limit': 1000,
    'noise': 0.05,
    'scale': (1.0, 1.0, 1.0),
    'margin': 0.00001,
}
with open(SETTINGS_PATH, 'r') as json_file:
    settings.update(**json.loads(json_file.read()))


# LOAD OBJECTS TO FRACTURE FROM .BLEND...
with bpy.data.libraries.load(SRC_PATH) as (data_from, data_to):
    data_to.objects = data_from.objects
to_export_objects: List[Object] = []
# b3d_data_path = SRC_PATH + "\\Object\\"
for ob_name in object_names:
    '''
    bpy.ops.wm.append(
        filepath=(b3d_data_path + ob_name),
        directory=b3d_data_path,
        filename=ob_name
    )
    '''
    ob = bpy.data.objects[ob_name]
    to_export_objects.append(ob)
    context.scene.collection.objects.link(ob)
    ob.select_set(False)


# CLIENT.
class SocketSignal:
    STARTED = 0
    CONTINUE = 123
    PROGRESS = 100
    FINISHED = 1
    ERROR = 666
    WAIT = 111
    REQUEST_SEND = 99

def rcv_signal(client: socket.SocketType) -> Union[int, None]:
    # Get data from connection.
    data = client.recv(struct.calcsize('i'))

    # Small delay if no data was found.
    if data is None:
        return None

    try:
        # Get signal.
        unpacked_data = struct.unpack('i', data)
        return unpacked_data[0]
    except:
        return None

def send_signal(client: socket.SocketType, signal: SocketSignal) -> None:
    packed_data = struct.pack('i i', instance_uid, signal)
    client.sendall(packed_data)


def create_collection(context, name: str) -> Collection:

    def get_layer_coll(layer_coll, collection):
        if (layer_coll.collection == collection):
            return layer_coll
        for layer in layer_coll.children:
            layer_coll = get_layer_coll(layer, collection)
            if layer_coll:
                return layer_coll
        return None

    # Create collection for object.
    coll_name: str = name # + '_low'
    collection = bpy.data.collections.new(coll_name)

    # Link new collection top scene collection.
    # BUG. RuntimeError: Error: Collection 'Whathever' already in collection 'Scene Collection'
    # context.scene.collection.children.link(collection)

    # Get layer collection from new collection.
    # Set as active.
    #act_layer_coll = context.view_layer.layer_collection
    #target_layer_coll = get_layer_coll(act_layer_coll, collection)
    #context.view_layer.active_layer_collection = target_layer_coll

    return collection


# FRACTURE PROCESS.
def fracture(context, to_fracture_ob: Object) -> Collection:
    # Ensure object is active and selected.
    context.view_layer.objects.active = to_fracture_ob
    to_fracture_ob.select_set(True)

    # Create a fracture collection and ensure is active.
    collection = create_collection(context, to_fracture_ob.name)

    context.scene.collection.children.link(collection)

    if len(to_fracture_ob.material_slots) == 0:
        bpy.ops.object.material_slot_add()
        to_fracture_ob.material_slots[0].material = bpy.data.materials.new('Mat__' + to_fracture_ob.name)

    # Fracture.
    bpy.ops.object.add_fracture_cell_objects(
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

    # Avoid original object to be selected nor active.
    to_fracture_ob.select_set(False)
    context.view_layer.objects.active = context.selected_objects[-1]

    context.scene.collection.children.unlink(collection)

    return collection # Output collection.


# LOOP.
with context.temp_override(**override_ctx), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect(('localhost', port))
    print("[Client-%i] Connected." % instance_uid, client.getsockname())

    output_collections: Set[Collection] = set()
    for ob in to_export_objects:
        print("[Client-%i] Fracturing Object... %s" % (instance_uid, ob.name))
        output_collections.add(fracture(context, ob))

    bpy.data.libraries.write(DST_PATH, output_collections)

    # SEND SIGNAL OF FINSIHED.
    print("[Client-%i] Done." % instance_uid)
    send_signal(client, SocketSignal.FINISHED)
