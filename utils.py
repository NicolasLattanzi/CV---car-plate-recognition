import numpy as np
import cv2


def vertices_from_image_path(path: str):
    # extraction of car plate vertices from image path/ name
    # example of a string of vertices: -386&473_177&454_154&383_363&402-
    raw_vertices = path.split('-')

    if len(raw_vertices) <= 4: return [0,0,0,0,0,0,0,0] # error prevention
    else: raw_vertices = raw_vertices[3]

    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: vertices += x.split('&')
    return list(map(int, vertices))


#implementare funzione per matrice di trasformazione 

def LP_photo(image, pts):
    # image resizing to 94x24
    dst_pts=np.array([ [0,0],[93,0],[93,23],[0,23] ], dtype="float32")

    Matrix = cv2.getPerspectiveTransform(np.array(pts, dtype="float32"), dst_pts)
    warped = cv2.warpPerspective(image, Matrix, (94, 24))
    return warped
