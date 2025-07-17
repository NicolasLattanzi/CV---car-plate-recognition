import os
import cv2
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional
import torch 
from PIL import Image

root="CCPD2019"
output_dir="crop photo/img"


def vertices_from_image_path(path: str):
    # extraction of car plate vertices from image path/ name
    raw_vertices = path.split('-')[3]
    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: vertices += x.split('&')
    return list(map(int, vertices))

def crop_photo(image, vertices):

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().numpy()

    
    vertices = list(map(int, vertices))

    # bottom-right, bottom-left, top-left, top-right
    pts = [ [vertices[i], vertices[i+1]] for i in range(0, 8, 2) ]

    # top-left, top-right, bottom-right, bottom-left
    pts = pts[2:] + pts[:2] # swap    

    height_padding = int(abs( pts[0][1] - pts[1][1] ))
    width_padding = int(abs( pts[0][0] - pts[3][0] ))
    cropped_height = pts[3][1] - pts[0][1] + 2*height_padding
    cropped_width = pts[1][0] - pts[0][0] + 2*width_padding

    cropped_img = functional.crop( img = image, 
                            top = pts[0][1] - height_padding, 
                            left = pts[0][0] - width_padding, 
                            height = cropped_height, 
                            width = cropped_width )
    
    transform = transforms.Resize((24, 94))
    return transform(cropped_img)


folders = [ 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather' ]
images = []
for folder_name in folders:
    folder_path = os.path.join(root, folder_name)
    for img in os.listdir(folder_path):
        if img.endswith('.jpg'):images.append( os.path.join(folder_path, img) )
for image in images:
    raw_vertices = image.split('-')[3]
    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: 
        vertices += x.split('&')
        list(map(int, vertices))
    img=Image.open(image)
    test=crop_photo(img, vertices)
    test = np.array(test)

    # Percorso di output (assicurati che la cartella esista)
    output_path = os.path.join("crop photo/img", os.path.basename(image))
    cv2.imwrite(output_path, test)




