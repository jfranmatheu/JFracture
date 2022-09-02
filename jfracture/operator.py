import functools
import json
from socket import SocketType
import subprocess
import sys
from tempfile import TemporaryFile
from time import sleep, time
from typing import List, Set, Tuple
from os import cpu_count, path, remove
from _thread import start_new_thread
import bpy
from bpy.types import Operator, Context, Object

from .server import JFractureServer, SocketSignal


CPU_COUNT = cpu_count()
MODULE_PATH = path.dirname(path.abspath(__file__))
SCRIPT_PATH = path.join(MODULE_PATH, 'script.py')
SETTINGS_PATH = path.join(MODULE_PATH, 'settings.json')
BLEND_PATH = path.join(MODULE_PATH, 'empty.blend')


def get_layer_coll(layer_coll, collection):
    if (layer_coll.collection == collection):
        return layer_coll
    for layer in layer_coll.children:
        layer_coll = get_layer_coll(layer, collection)
        if layer_coll:
            return layer_coll
    return None


class JFRACTURE_OT_cell_fracture(Operator):
    bl_idname: str = 'jfracture.cell_fracture'
    bl_label: str = "Cell Fracture"
    bl_description: str = "Multi-Thread based cell fracture"


    def init(self, context) -> None:
        self.start_time = time()
        self.timer = None

        # Write cf props to settings json.
        data = {}
        with open(SETTINGS_PATH, 'w') as json_file:
            json_string = json.dumps(data)
            json_file.write(json_string)

        # Deselect objects.
        for ob in self.objects:
            ob.select_set(False)
        self.iter_objects = iter(self.objects)
        self.iter_index = 0

        # Start server.
        connection_count: int = min(CPU_COUNT, len(self.objects))
        self.server = JFractureServer(connection_count)
        self.server.start()
        self.current_connection_count: int = 0
        self.max_connections = connection_count

        # Start clients.
        self.timer_func = functools.partial(self.start_clients, self.server)
        bpy.app.timers.register(self.timer_func)

        # Inter-Client properties.
        self.prev_instance_is_ready: bool = True
        self.waiting_instance: int = -1
        self.finished_instances_count: int = 0
        self.finished = [False] * len(self.objects)
        self.ready = [False] * len(self.objects)

        self.request_queue = []


    def start_clients(self, server: 'JFractureServer'):
        #idx = 0
        if self.current_connection_count >= self.max_connections:
            return None
        if self.iter_index >= len(self.objects):
            return None
        if not self.prev_instance_is_ready:
            return 0.2
        self.start_instance(bpy.context)
        try:
            client, address = server.new_connection()
        except Exception as e:
            print(e)
            return None
        print('[Server] Connected to: ' + address[0] + ':' + str(address[1]))
        start_new_thread(self.client_handler, (client, )) #idx
        #idx += 1
        self.current_connection_count += 1
        self.prev_instance_is_ready = False
        return 0.1


    def error(self, msg: str) -> None:
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}


    def finish(self, context: Context) -> None:
        self.server.stop()

        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None

        if bpy.app.timers.is_registered(self.timer_func):
            bpy.app.timers.unregister(self.timer_func)
            del self.timer_func

        tot_time = time() - self.start_time
        time_msg = "Total Time: %.2f" % (tot_time)
        self.report({'INFO'}, time_msg)
        print(time_msg)
        return {'FINISHED'}


    def execute(self, context: Context) -> Set[str]:
        if CPU_COUNT == 0:
            return self.error("No CPU! Please, buy new computer")

        if len(context.selected_objects) == 0:
            return self.error("No Selected Objects!")

        filt_objects: List[Object] = [ob for ob in context.selected_objects if ob.type=='MESH' and ob.visible_get()]
        if not filt_objects:
            return self.error("No Selected MESH Objects!")

        #if any([bool(ob.modifiers) for ob in filt_objects]):
        #    return self.error("Please, apply or remove modifiers from selected objects")

        self.objects = filt_objects
        self.init(context)

        self.timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


    def modal(self, context: Context, event) -> Set[str]:
        #print(event.type, event.value)
        if not event.type.startswith('TIMER'):
            return {'RUNNING_MODAL'}

        # Workaround to safely resolve the client requests without conflicts between
        # the different threads that hold the connections with the clients.
        while self.request_queue != []:
            client_id, request, client = self.request_queue.pop(0)
            if request == SocketSignal.REQUEST_SEND:
                if self.waiting_instance != -1:
                    # Already waiting for an instance to finish.
                    print("[Server] Instance { %i } requested to send but Waiting for instance { %i } to finish..." % (client_id, self.waiting_instance))
                    self.server.send_signal(client, SocketSignal.WAIT)
                    continue
                print("[Server] Instance { %i } requested to send, the request is granted!" % client_id)
                self.waiting_instance = client_id
                self.server.send_signal(client, SocketSignal.CONTINUE)
                return {'RUNNING_MODAL'}

            elif request == SocketSignal.FINISHED:
                if self.waiting_instance != client_id:
                    print("[Server] WTF this client want to finish but it should not finish nor continue!!!", client_id)
                print("[Server] Instance { %i } just finished!" % client_id)
                self.finished[client_id] = True

        if self.finished.count(True) == self.finished_instances_count:
            return {'RUNNING_MODAL'}

        self.finished_instances_count = self.finished.count(True)

        index = self.waiting_instance
        fractured_ob: Object = self.objects[index]

        # Create collection for object.
        coll_name: str = fractured_ob.name + '_low'
        collection = bpy.data.collections.new(coll_name)

        # Link new collection top scene collection.
        context.scene.collection.children.link(collection)

        # Get layer collection from new collection.
        # Set as active.
        act_layer_coll = context.view_layer.layer_collection
        target_layer_coll = get_layer_coll(act_layer_coll, collection)
        context.view_layer.active_layer_collection = target_layer_coll

        # Paste back from clipboard buffer.
        bpy.ops.view3d.pastebuffer(False, active_collection=True, autoselect=False)

        print("[Server] Info! Object { %s } was successfully fractured into %i chunks! (by Client-%i)" % (fractured_ob.name, len(context.selected_objects), index))

        self.waiting_instance = -1 # RESET.
        self.current_connection_count -= 1

        if all(self.finished):
            print("[Server] DONE!")
            return self.finish(context)

        return {'RUNNING_MODAL'}


    def client_handler(self, client: SocketType): #, client_id: int):
        print('[Server] Started New Thread.')
        with client:
            while 1:
                #print("holiwoli")
                try:
                    data: Tuple[int, int] = self.server.rcv_signal(client)
                    if data is None:
                        continue

                    instance_id, signal = data
                    index = instance_id

                    print(f"[Server] Received signal {signal} from instance {instance_id}")

                    if signal == SocketSignal.STARTED:
                        self.ready[index] = True
                        self.prev_instance_is_ready = True
                        continue

                    if not all(self.ready):
                        # If not all instances are ready... break.
                        if signal == SocketSignal.REQUEST_SEND:
                            # Just in case, some fast guy want to make some trouble.
                            self.server.send_signal(client, SocketSignal.WAIT)
                        sleep(0.1)
                        continue

                    """
                    if signal == SocketSignal.REQUEST_SEND:
                        # Nice but unsafe.
                        if self.waiting_instance != -1:
                            # Already waiting for an instance to finish.
                            #print("Waiting...")
                            print("[Server] Instance { %i } requested to send but Waiting for instance { %i } to finish..." % (instance_id, self.waiting_instance))
                            self.server.send_signal(client, SocketSignal.WAIT)
                            continue
                        print("[Server] Instance { %i } requested to send, the request is granted!" % instance_id)
                        self.waiting_instance = instance_id
                        self.server.send_signal(client, SocketSignal.CONTINUE)

                    elif signal == SocketSignal.FINISHED:
                        print("[Server] Instance { %i } just finished!" % instance_id)
                        self.finished[index] = True

                        # Disconnect client.
                        #client.close() # automatic with "with client"
                        break
                    """
                    # Safe method.
                    self.request_queue.append((instance_id, signal, client))
                    sleep(0.1)
                    continue

                except Exception as e:
                    print(e)
                    break

    def start_instance(self, context) -> None:
        try:
            ob = next(self.iter_objects)
        except StopIteration:
            print("[Server] Info! No more objects to fracture left...")
            return

        #print(ob)

        # Select/Set Active object.
        context.view_layer.objects.active = ob
        ob.select_set(True)

        # Copy object to clipboard buffer.
        bpy.ops.view3d.copybuffer(False)

        # Start instance.
        process = subprocess.Popen(
            [
                bpy.app.binary_path, # sys.executable,
                BLEND_PATH,
                '--background',
                '--python',
                SCRIPT_PATH,
                '--', # Blender is silly and stops with this.
                # Now the arguments...
                str(self.iter_index), # ID.
                str(self.server.port),
                #self.temp_file
            ],
            shell=False)

        # Deselect.
        ob.select_set(False)
        ob.hide_set(True)

        self.iter_index += 1

        print("[Server] Starting Up new instance { %i } for object { %s }" % (self.iter_index, ob.name))

        #sleep(.1)
