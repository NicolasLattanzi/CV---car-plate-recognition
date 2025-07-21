from torchvision import transforms
from torch.utils.data import *
from PIL import Image

import torch
import numpy as np
import cv2
import os

import utils

class CarPlateDataset(Dataset):

    def __init__(self, path, training=True):
        self.root = path
        if training:
            full_path = os.path.join(self.root, 'ccpd_base')
            self.images = [os.path.join(full_path, img) for img in os.listdir(full_path) if img.endswith('.jpg')]
        else: # evaluation
            folders = [ 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather' ]
            self.images = []
            for folder_name in folders:
                folder_path = os.path.join(self.root, folder_name)
                for img in os.listdir(folder_path):
                    if img.endswith('.jpg'): self.images.append( os.path.join(folder_path, img) )
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        v1,v2,v3,v4,v5,v6,v7,v8 = utils.vertices_from_image_path(img_path)
        w, h = image.size
        vertices = torch.tensor( [ v1/w, v2/h, v3/w, v4/h, v5/w, v6/h, v7/w, v8/h ], dtype=torch.float32 ) # normalization
        license_plate = utils.plate_from_image_path(img_path)
        license_plate = torch.tensor( license_plate, dtype=torch.int32 )

        if self.transform:
            image = self.transform(image)

        return image, vertices, license_plate

def train_test_split(dataset, train=0.5, test=0.5):
    return torch.utils.data.random_split(dataset, [train, test])


# create a database of cropped photos around the license plates.
# used to train/ test the recognition model
def create_cropped_dataset():
    folders = [ 'ccpd_blur', 'ccpd_challenge', 'ccpd_db', 'ccpd_fn', 'ccpd_np', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_weather' ]
    images = []
    output_dir="crop photo/img"

    for folder_name in folders:
        folder_path = os.path.join("CCPD2019", folder_name)
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
        test=utils.crop_photo(img, vertices)
        test = np.array(test)

        # Percorso di output (assicurati che la cartella esista)
        output_path = os.path.join(output_dir, os.path.basename(image))
        cv2.imwrite(output_path, test)

class LicensePlateDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_list = [fname for fname in os.listdir(img_dir)if fname.endswith('.jpg') and len(fname.split('_')) == 7]  # sanity check (opzionale)
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        fname=self.img_list[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_str = os.path.splitext(fname)[0]
        label = [int(x) for x in label_str.split('_')]
        label_tensor = torch.tensor(label, dtype=torch.long)
        target_length=len(label_tensor)
        return image, label_tensor, target_length
        
        

