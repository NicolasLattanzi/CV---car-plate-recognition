import numpy as np
import cv2
from torchvision import transforms


def vertices_from_image_path(path: str):
    # extraction of car plate vertices from image path/ name
    # example of a string of vertices: -386&473_177&454-
    raw_vertices = path.split('-')[2]
    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: vertices += x.split('&')
    return list(map(int, vertices))


#implementare funzione per matrice di trasformazione 

def LP_photo(image, pts):
    dst_pts=np.array([ [93,23],[0,23],[0,0],[93,0] ], dtype="float32")

    Matrix = cv2.getPerspectiveTransform(np.array(pts, dtype="float32"), dst_pts)
    warped = cv2.warpPerspective(image, Matrix, (94, 24))
    rgbConv=BgrToRgb(warped)#converto l'immagine da bgr a rgb
    transform=transforms.Compose([transforms.ToTensor()])  #convertitore in tensore
    tensorTrans=transform(rgbConv).unsqueeze(0)     #converto l'immagine in un tensore per darlo a LPRNet
    return tensorTrans

def BgrToRgb(image):
    rgbConv=cv2.cvtColor(image, cv2.COLORBGR2RGB)
    return rgbConv
