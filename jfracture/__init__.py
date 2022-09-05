# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

bl_info = {
    "name" : "jfracture",
    "author" : "JF racture",
    "description" : "Fast Cell Fracture",
    "blender" : (3, 2, 0),
    "version" : (0, 6, 0),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

def install_deps():
    import sys

    from . import pyhull
    sys.modules['pyhull'] = pyhull

install_deps()

def register():
    from bpy.utils import register_class
    from .operator import JFRACTURE_OT_cell_fracture
    register_class(JFRACTURE_OT_cell_fracture)

def unregister():
    from bpy.utils import unregister_class
    from .operator import JFRACTURE_OT_cell_fracture
    unregister_class(JFRACTURE_OT_cell_fracture)
