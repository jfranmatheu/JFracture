import json
from socket import SocketType
import subprocess
from time import sleep, time
from typing import List, Set, Tuple
from os import cpu_count, path
from _thread import start_new_thread
import bpy
from bpy.types import Operator, Context, Object
from tempfile import gettempdir
from threading import Thread

from .server import JFractureServer, SocketSignal


CPU_COUNT = cpu_count()
MODULE_PATH = path.dirname(path.abspath(__file__))
SCRIPT_PATH = path.join(MODULE_PATH, 'script.py')
SETTINGS_PATH = path.join(MODULE_PATH, 'settings.json')
BLEND_PATH = path.join(MODULE_PATH, 'empty.blend')
TMP_DIR = gettempdir()
COPYBUFFER_PATH = path.join(TMP_DIR, 'coppybuffer.blend')


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_layer_coll(layer_coll, collection):
    if (layer_coll.collection == collection):
        return layer_coll
    for layer in layer_coll.children:
        layer_coll = get_layer_coll(layer, collection)
        if layer_coll:
            return layer_coll
    return None


def get_cell_count(ob: Object) -> int:
    count: int = 0
    for ps in ob.particle_systems:
        count += ps.settings.count
    return count


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
            # Find aprox number of particles/cells.
            ob['cell_count'] = get_cell_count(ob)
            ob['fracture_cost'] = len(ob.data.vertices) / ob['cell_count']

        # Get instance count.
        ob_count: int = len(self.objects)
        instances_count: int = min(CPU_COUNT, ob_count)
        self.instances_count = instances_count

        # Sort objects to find best efficiency.
        # k: amount of completed instances (each one with an equivalent amount of objects).
        # m: amount of objects rest that can't fill an instance (completely).
        # k, m = divmod(ob_count, instances_count)
        # Sort by cell count.
        self.objects.sort(key=lambda ob: ob['fracture_cost'], reverse=True)
        # print(self.objects)
        instances_cost: List[int] = [0] * instances_count
        # sliced_ob_list = self.objects[:-m] if m != 0 else self.objects
        instance_objects: List[List[Object]] = []
        for _i in range(0, instances_count):
            instance_objects.append([])
        for ob in self.objects:
            print(min(instances_cost))
            index_min = instances_cost.index(min(instances_cost))
            print(index_min)
            instance_objects[index_min].append(ob)
            print(ob)
            instances_cost[index_min] += ob['fracture_cost']
        # print(instances_cost)
        # print(instance_objects)

        '''
        # OLD. Linear, not smart.
        # Resolve objects per instance.
        if instances_count == ob_count:
            instance_objects = [[ob] for ob in self.objects]
        else:
            # ob_count > instances_count
            instance_objects: List[List[Object]] = split(self.objects, instances_count)
        '''

        # Inter-Client properties.
        self.finished = [False] * instances_count
        self.request_queue = []

        # Export objects.
        write_lib = bpy.data.libraries.write
        instance_object_names = []
        for idx, objects in enumerate(instance_objects):
            output_path: str = path.join(TMP_DIR, 'coppybuffer' + str(idx) + '.blend')
            write_lib(output_path, set(objects))
            names = []
            for ob in objects:
                names.append(ob.name)
                bpy.data.objects.remove(ob)
            instance_object_names.append(names)
        del instance_objects

        # Iterator.
        self.iter_instance_objects = iter(instance_object_names)
        self.iter_index: int = 0

        # Start server.
        self.server = JFractureServer(instances_count)
        self.server.start()

        # Start clients.
        self.thread = Thread(target=self.start_clients, name="Client Initializer", daemon=True)
        self.thread.start()

        return True


    def start_clients(self):
        server: JFractureServer = self.server
        while 1:
            if self.iter_index >= self.instances_count:
                return None
            try:
                object_names = next(self.iter_instance_objects)
            except StopIteration:
                print("[Server] Info! No more objects left to fracture.")
                return None
            self.start_instance(object_names, str(self.iter_index))
            try:
                client, address = server.new_connection()
            except Exception as e:
                print(e)
                return None
            print('[Server] Connected to: ' + address[0] + ':' + str(address[1]))
            start_new_thread(self.client_handler, (client, )) #idx
            self.iter_index += 1


    def start_instance(self, object_names: List[str], instance_id: str) -> None:
        print("[Client-%s] Initializing..." % instance_id)

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
                str(self.server.port),
                instance_id, # ID.
                ','.join(object_names)
            ],
            shell=False)


    def error(self, msg: str) -> None:
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}


    def finish(self, context: Context) -> None:
        self.server.stop()

        if self.timer:
            context.window_manager.event_timer_remove(self.timer)
            self.timer = None

        if self.thread:
            del self.thread

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

        self.objects = filt_objects
        if not self.init(context):
            return {'CANCELLED'}

        self.timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


    def modal(self, context: Context, event) -> Set[str]:
        if not event.type.startswith('TIMER'):
            return {'RUNNING_MODAL'}

        # Workaround to safely resolve the client requests without conflicts between
        # the different threads that hold the connections with the clients.
        while self.request_queue != []:
            client_id, request, client = self.request_queue.pop(0)
            if request == SocketSignal.FINISHED:
                print("[Server] Client-%i just finished!" % client_id)
                self.finished[client_id] = True
                lib_path = path.join(TMP_DIR, 'pastebuffer' + str(client_id) + '.blend')
                # Load output fracture collections from client output.
                with bpy.data.libraries.load(lib_path) as (data_from, data_to):
                    data_to.collections = data_from.collections
                link_coll = context.scene.collection.children.link
                for collection in data_to.collections:
                    if collection is not None:
                        link_coll(collection)
                print("[Server] Chunks generated by Client-%i were loaded successfully!" % client_id)
            else:
                print("Woot!?")

        if all(self.finished):
            print("[Server] DONE!")
            return self.finish(context)

        return {'RUNNING_MODAL'}


    def client_handler(self, client: SocketType):
        print('[Server] Started New Thread.')
        with client:
            while 1:
                try:
                    data: Tuple[int, int] = self.server.rcv_signal(client)
                    if data is None:
                        continue

                    instance_id, signal = data
                    print(f"[Server] Received signal {signal} from instance {instance_id}")

                    # Safe method.
                    self.request_queue.append((instance_id, signal, client))

                    if signal == SocketSignal.FINISHED:
                        break
                    sleep(0.1)
                    continue

                except Exception as e:
                    print(e)
                    break
