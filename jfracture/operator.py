from math import ceil
import subprocess
from time import time
from typing import List, Set, Union
from os import cpu_count, path
import bpy
from bpy.types import Operator, Context, Object
from tempfile import gettempdir
from threading import Thread


CPU_COUNT = cpu_count()
MODULE_PATH = path.dirname(path.abspath(__file__))

SCRIPT_PATH = path.join(MODULE_PATH, 'script.py')
# SCRIPT_PATH = path.join(MODULE_PATH, 'script_pyhull.py')

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
    bl_label: str = "JFracture Cell Fracture"
    bl_description: str = "Multi-Thread based cell fracture"
    bl_options = {'REGISTER', 'UNDO'}

    def init(self, context) -> None:
        self.start_time = time()

        # Write cf props to settings json.
        data = {}
        with open(SETTINGS_PATH, 'w') as json_file:
            import json
            json_string = json.dumps(data)
            json_file.write(json_string)

        # Deselect objects.
        small_fracture_objects: List[Object] = []
        bif_fracture_object_count: int = 0
        fracture_count: int = 0
        max_fractures_per_instance: int = 1000
        for ob in self.objects:
            ob.select_set(False)
            # Find aprox number of particles/cells.
            ob['cell_count'] = max(1, get_cell_count(ob))
            ob['fracture_cost'] = len(ob.data.vertices) / ob['cell_count']
            if ob['cell_count'] < 75:
                small_fracture_objects.append(ob)
            elif ob['cell_count'] > max_fractures_per_instance:
                bif_fracture_object_count += 1
            else:
                fracture_count += ob['cell_count']

        # Removed discarded objects.
        self.objects = [ob for ob in self.objects if ob['cell_count'] >= 75]
        self.small_fracture_objects = small_fracture_objects

        # Get instance count.
        ob_count: int = len(self.objects)
        # Half instance count if we detect that at least 50% objects have less than 500 fractures.
        instances_count: int = min(CPU_COUNT, ceil(fracture_count / max_fractures_per_instance) +
                                   bif_fracture_object_count)  # ob_count)
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
            # print(min(instances_cost))
            index_min = instances_cost.index(min(instances_cost))
            # print(index_min)
            instance_objects[index_min].append(ob)
            # print(ob)
            instances_cost[index_min] += ob['fracture_cost']
        # print(instances_cost)
        # print(instance_objects)

        # Start alternative fracturer for small fractures.
        if self.small_fracture_objects:
            small_ob_count: int = len(self.small_fracture_objects)
            while 1:
                if small_ob_count >= 100:
                    small_objects = self.small_fracture_objects[:50]
                    self.small_fracture_objects = self.small_fracture_objects[50:]
                else:
                    small_objects = list(self.small_fracture_objects)
                    self.small_fracture_objects.clear()

                self.instances_count += 1
                instance_objects.append(small_objects)
                small_ob_count = len(self.small_fracture_objects)
                if small_ob_count == 0:
                    break

        # Inter-Client properties.
        self.finished = [False] * self.instances_count
        self.client_processes = []

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

        # Start clients.
        self.thread = Thread(target=self.start_clients, name="Client Initializer", daemon=True)
        self.thread.start()

        return True

    def start_clients(self):
        while 1:
            if self.iter_index >= self.instances_count:
                return None
            try:
                object_names = next(self.iter_instance_objects)
            except StopIteration:
                print("[Server] Info! No more objects left to fracture.")
                return None
            self.start_instance(object_names, str(self.iter_index))
            self.iter_index += 1

    def start_instance(self, object_names: List[str], instance_id: str) -> None:
        print("[Client-%s] Initializing..." % instance_id)

        # Start instance.
        popen = subprocess.Popen(
            [
                bpy.app.binary_path,  # sys.executable,
                BLEND_PATH,
                '--background',
                '--debug',
                '--python',
                SCRIPT_PATH,
                '--',  # Blender is silly and stops with this.
                # Now the arguments...
                instance_id,  # ID.
                ','.join(object_names)
            ],
            shell=False)

        if popen:
            self.client_processes.append(popen)

    def error(self, msg: str) -> None:
        self.report({'ERROR'}, msg)
        return {'CANCELLED'}

    def finish(self, context: Context) -> None:
        if hasattr(self, 'timer'):
            context.window_manager.event_timer_remove(self.timer)
            del self.timer

        if hasattr(self, 'thread'):
            del self.thread

        if hasattr(self, 'client_processes'):
            del self.client_processes[:]

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

        filt_objects: List[Object] = [ob for ob in context.selected_objects if ob.type == 'MESH' and ob.visible_get()]
        if not filt_objects:
            return self.error("No Selected MESH Objects!")

        self.objects = filt_objects
        if not self.init(context):
            return {'CANCELLED'}
        if self.instances_count == 0:
            return self.finish(context)

        self.timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context: Context, event) -> Set[str]:
        if not event.type.startswith('TIMER'):
            return {'RUNNING_MODAL'}

        finished: List[int] = []
        for client_id, process in enumerate(self.client_processes):
            if process is None:
                continue
            ret: Union[int, None] = process.poll()
            if ret is None:
                continue
            if ret == 0:
                print("[Server] Client-%i just finished!" % client_id)
                print(self.client_processes[client_id])
                self.finished[client_id] = True
                lib_path = path.join(TMP_DIR, 'pastebuffer' + str(client_id) + '.blend')
                # Load output fracture collections from client output.
                if path.isfile(lib_path):
                    with bpy.data.libraries.load(lib_path) as (data_from, data_to):
                        data_to.collections = data_from.collections
                    link_coll = context.scene.collection.children.link
                    for collection in data_to.collections:
                        if collection is not None:
                            link_coll(collection)
                    finished.append(client_id)
                    print("[Server] Chunks generated by Client-%i were loaded successfully!" % client_id)
                else:
                    print(lib_path, 'Dosent exist!')
            else:
                print("[Server] ERROR! Client-%i failed!" % client_id)

        if finished:
            for client_id in finished:
                self.client_processes[client_id] = None

        if all(self.finished):
            print("[Server] DONE!")
            return self.finish(context)

        return {'RUNNING_MODAL'}
