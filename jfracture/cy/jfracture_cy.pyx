from libc.stdio cimport printf
from libcpp cimport bool, float

import numpy as np
cimport numpy as np
cimport cython

np.import_array()

import bpy
from mathutils.geometry import points_in_planes
from math import sqrt


''' Structures. '''
ctypedef struct MVert:
    float co[3]
    # 3.1 change: https://github.com/blender/blender/commit/cfa53e0fbeed7178c7876413e2010fd3347d7f72
    # short no[3] # Now it is accesible directly from the mesh itself. type => const float (*vert_normals)[3];
    char flag
    char bweight

ctypedef struct MPoly:
    int loopstart
    int totloop
    short mat_nr # Material slot index?
    char flag
    char _pad

ctypedef struct MLoop:
    unsigned int v # Vertex index.
    unsigned int e # Edge index.

ctypedef struct MLoopUV:
    float uv[2]
    int flag

ctypedef struct MLoopCol:
    unsigned char r
    unsigned char g
    unsigned char b
    unsigned char a

ctypedef struct MLoopTri:
    unsigned int tri[3]
    unsigned int poly

ctypedef struct MVertTri:
    unsigned int tri[3]


''' Functions. '''

def sort_key(p):
    global _point_cell_current
    return np.sqrt((p-_point_cell_current))**2

def get_sorted(arr):
    dots = np.array(p.dot(p) for p in arr)
    sorted_indices = np.argsort(dots, axis=0)
    return arr[sorted_indices]

def e_dist(a, b, metric='euclidean'):
    """Distance calculation for 1D, 2D and 3D points using einsum
    : a, b   - list, tuple, array in 1,2 or 3D form
    : metric - euclidean ('e','eu'...), sqeuclidean ('s','sq'...),
    :-----------------------------------------------------------------------
    """
    a = np.asarray(a)
    b = np.atleast_2d(b)
    a_dim = a.ndim
    b_dim = b.ndim
    if a_dim == 1:
        a = a.reshape(1, 1, a.shape[0])
    if a_dim >= 2:
        a = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    if b_dim > 2:
        b = b.reshape(np.prod(b.shape[:-1]), b.shape[-1])
    diff = a - b
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    if metric[:1] == 'e':
        dist_arr = np.sqrt(dist_arr)
    dist_arr = np.squeeze(dist_arr)
    return dist_arr

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef list get_cells(
    np.ndarray[np.float32_t, ndim=2] convex_planes,
    np.ndarray[np.float32_t, ndim=2] points,
    float margin):

    cdef int point_count = len(points)

    #cdef np.ndarray[np.float32_t, ndim=2] cells = np.empty([1,1], dtype=np.float32)
    cells = []

    cdef np.ndarray[np.float32_t, ndim=1] point_cell_current # = points[0]
    cdef np.ndarray[np.float32_t, ndim=2] planes # = convex_planes.copy()
    cdef np.ndarray[np.float32_t, ndim=2] points_sorted_current = points.copy()
    #points_sorted_current = [*points]

    cdef np.ndarray[np.float32_t, ndim=1] plane = np.zeros((4), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] normal
    cdef float nlength = 0
    cdef float distance_max = 10000000000.0  # a big value!

    vertices = []
    plane_indices = []

    for index in range(point_count):
        point_cell_current = points[index]
        #global _point_cell_current
        #_point_cell_current = point_cell_current
        planes = convex_planes.copy()
        for j in range(6):
            for i in range(3):
                planes[j][3] += planes[j][i]*point_cell_current[i]

        arr = points_sorted_current-point_cell_current
        dots = np.empty((len(arr)), dtype=np.float32)
        for i in range(len(arr)):
            dots[i] = np.array(arr[i].dot(arr[i]))
        sorted_indices = np.argsort(dots)
        points_sorted_current = points_sorted_current[sorted_indices] # get_sorted((points_sorted_current-point_cell_current))

        #points_sorted_current = e_dist(points_sorted_current, point_cell_current.copy().reshape((1, 3)))
        #newList = points_sorted_current - point_cell_current
        #sort = np.sqrt((newList**2))
        #points_sorted_current = newList[sort.argmin()]
        #points_sorted_current = get_sorted(points_sorted_current, point_cell_current)
        #points_sorted_current = arr[
        #    np.lexsort(keys=[
        #        np.vectorize(lambda p: np.sqrt((p-point_cell_current)**2))(points_sorted_current)
        #    ])
        #]
        # points_sorted_current = sorted(points_sorted_current, key=lambda p: np.sqrt((p-point_cell_current)**2))
        #points_sorted_current = sorted(points_sorted_current, key=sort_key)
        # points_sorted_current = np.array(list(points_sorted_current).sort(key=sort_key), dtype=np.float32)
        distance_max = 10000000000.0

        for j in range(1, point_count):
            normal = points_sorted_current[j] - point_cell_current
            nlength = np.sqrt(normal.dot(normal)) # np.linalg.norm(normal)

            if nlength > distance_max:
                break

            for i in range(3):
                plane[i] = normal[i] / nlength # np.linalg.norm(normal)
            plane[3] = (-nlength / 2.0) + margin

            planes += plane
            '''
            vertices[:], plane_indices[:] = points_in_planes(planes)
            if len(vertices) == 0:
                break

            if len(plane_indices) != len(planes):
                planes[:] = [planes[k] for k in plane_indices]

            distance_max = sqrt(max(v.length_squared for v in vertices)) * 2.0  # sqrt -> make real length
            '''

            vertices[:], plane_indices[:] = points_in_planes(planes)
            if len(vertices) == 0:
                break

            if len(plane_indices) != len(planes):
                planes = np.array([planes[k] for k in plane_indices])

            distance_max = max([v.length_squared for v in vertices]) # v.dot(v)
            distance_max = sqrt(distance_max) * 2.0

        if len(vertices) < 4:
            continue

        cells.append((point_cell_current, [*vertices]))

    return cells
