import json
import struct
from time import sleep
from typing import Union
import bpy
from bpy import context

import sys
import socket
from os.path import join, dirname, abspath


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
    '0',
    '64387'
]
'''
# Python.exe --- sys.argv[0]
# print("Args:", sys.argv)
instance_uid: int = int(sys.argv[-2]) # Un identificador numérico único.
port: int = int(sys.argv[-1]) # Puerto de comunicación con el server.


MODULE_DIR = dirname(abspath(__file__))
SETTINGS_PATH = join(MODULE_DIR, 'settings.json')
#TEMP_FILE = sys.argv[-1]


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
print(MODULE_DIR)
print("SETTINGS PATH:", SETTINGS_PATH)
with open(SETTINGS_PATH, 'r') as json_file:
    settings.update(**json.loads(json_file.read()))

# CLIENT.
class SocketSignal:
    STARTED = 0
    CONTINUE = 123
    PROGRESS = 100
    FINISHED = 1
    ERROR = 666
    WAIT = 111
    REQUEST_SEND = 99

'''
class JFractureClient:
    address_family: socket.AddressFamily = socket.AF_INET
    socket_type: socket.SocketKind = socket.SOCK_STREAM

    def __init__(self) -> None:
        self.socket: socket.SocketType = socket.socket(
            self.address_family, self.socket_type)
        self.server_address: Tuple[str, int] = ('localhost', port)

    def __enter__(self) -> 'JFractureClient':
        self.start()
        return self if self.socket else None

    def __exit__(self, *args) -> None:
        self.stop()

    def start(self) -> int:
        # Wait for it to respond.
        print('[SOCKET][CLIENT] Starting up on {} port {}'.format(*self.server_address))
        try:
            self.socket.settimeout(5)
            self.socket.connect(self.server_address)
            self.socket.settimeout(None)
        except (socket.timeout, ConnectionRefusedError) as e:
            # Connection is dropped!
            self.socket = None
            self.error = e.strerror
            print(e)
            return 1
        return 0

    def send_signal(self, signal: SocketSignal):
        packed_data: bytes = struct.pack('i', signal)
        self.socket.send(packed_data)
        packed_data: bytes = struct.pack('i', instance_uid)
        self.socket.send(packed_data)

    def stop(self):
        if self.socket:
            self.socket.close()
            self.socket = None
'''

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


# FRACTURE PROCESS.
with context.temp_override(**override_ctx), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect(('localhost', port))
    print("[Client-%i] Connected." % instance_uid, client.getsockname())

    # Paste from clipboard buffer.
    bpy.ops.view3d.pastebuffer(autoselect=True, active_collection=True)

    # Notify server that the process just started.
    print("[Client-%i] Started." % instance_uid)
    send_signal(client, SocketSignal.STARTED)

    #print(list(context.selected_objects))
    #print(list(context.view_layer.objects))

    to_fracture_ob = context.selected_objects[-1]
    context.view_layer.objects.active = to_fracture_ob

    if len(to_fracture_ob.material_slots) == 0:
        bpy.ops.object.material_slot_add({'object': to_fracture_ob})
        to_fracture_ob.material_slots[0].material = bpy.data.materials.new('kk')

    print("[Client-%i] Fracturing Object... %s" % (instance_uid, to_fracture_ob.name))
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
        #collection_name=coll_name
    )

    # Avoid original object to be selected nor active.
    to_fracture_ob.select_set(False)
    context.view_layer.objects.active = context.selected_objects[-1]

    # SEND SIGNAL OF FINSIHED.
    while 1:
        try:
            print("[Client-%i] Requesting send data to server." % instance_uid)
            send_signal(client, SocketSignal.REQUEST_SEND)
            print("[Client-%i]\t- Trying to receive an answer from server" % instance_uid)
            signal = rcv_signal(client)
            #print(f"Client {instance_uid} rcv signal {signal}")
            if signal == SocketSignal.CONTINUE:
                print("[Client-%i]\t- Continue!" % instance_uid)
                break
            elif signal == SocketSignal.WAIT:
                print("[Client-%i]\t- Waiting..." % instance_uid)
            sleep(0.2)
        except ConnectionAbortedError as e:
            print(e, " - instance ->", instance_uid)
            sleep(1.0)

    # Copy back to clipboard buffer.
    bpy.ops.view3d.copybuffer()

    # SEND SIGNAL OF FINSIHED.
    print("[Client-%i] Done." % instance_uid)
    send_signal(client, SocketSignal.FINISHED)
