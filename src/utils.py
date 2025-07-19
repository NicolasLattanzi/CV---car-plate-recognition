import torch
from torchvision import transforms
from torchvision.transforms import functional
import torch.nn.functional as f
import cv2
import numpy as np

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", 
             "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", 
             "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def vertices_from_image_path(path: str):
    # extraction of car plate vertices from image path/ name
    raw_vertices = path.split('-')[3]
    raw_vertices = raw_vertices.split('_')
    vertices = []
    for x in raw_vertices: vertices += x.split('&')
    return list(map(int, vertices))

def plate_from_image_path(path: str):
    # extraction of car license plate from image path/ name
    plate = path.split('-')[4]
    plate_chars = plate.split('_')
    # ''.join(plate_chars)
    return list(map(int, plate_chars))



def crop_photo(image, vertices):

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().numpy()

    w = h = 224
    v1,v2,v3,v4,v5,v6,v7,v8 = vertices
    vertices = [ v1*w,v2*h,v3*w,v4*h,v5*w,v6*h,v7*w,v8*h ] # denormalization
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


def lpDecoder(license,pr_list=provinces, char_list=ads):
    
    decoded=[]
    print(license)
    for idx in license.squeeze():
        if(len(decoded)==0):
            decoded.append(pr_list[idx])
        else:
            decoded.append(char_list[idx])
    return ''.join(decoded)
    
