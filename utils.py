import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.transforms import functional, InterpolationMode


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
    pts = [ [vertices[i], vertices[i+1]] for i in range(0, 8, 2) ]

    # order must be:             top-left, top-right, bottom-right, bottom-left
    # but vertices are in order: bottom-right, bottom-left, top-left, top-right
    pts = pts[4:] + pts[:4] # swap
    output_size = (94, 24)
    dst_pts = [[0, 0], [output_size[0]-1, 0], [output_size[0]-1, output_size[1]-1], [0, output_size[1]-1]]

    cropped_img = functional.perspective(image, startpoints=pts, endpoints=dst_pts, interpolation=InterpolationMode.BILINEAR)

    return cropped_img


