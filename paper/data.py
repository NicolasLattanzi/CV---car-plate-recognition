from tqdm import tqdm
from PIL import Image
import random
import os

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", 
             "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", 
             "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

CHARS = [
    '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣',
    '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    '新', '警', '学', 'O',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q',
    'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'O',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'
]
def extract_info_from_path(path):
    parts = path.split('-')
    vertices = parts[2]
    lp = parts[4]
    
    # vertices extraction
    x1, y1 = map(int, vertices.split('_')[0].split('&'))
    x2, y2 = map(int, vertices.split('_')[1].split('&'))

    # label chars extraction
    label_parts = lp.split('_')
    indices = list(map(int, label_parts))
    chars = [provinces[indices[0]], alphabets[indices[1]]]
    for i in range(2, 7):
        chars.append(ads[indices[i]])
    label = ''.join(chars)

    return (x1, y1, x2, y2), label

def create_subfolder(images, img_dir, label_file):
    with open(label_file, 'w', encoding='utf-8') as f:
        for i, img in enumerate(tqdm(images)):
            vertices, label = extract_info_from_path(img)
            img_path = os.path.join(ccpd_dir, img)
            img = Image.open(img_path)

            crop = img.crop(vertices)
            crop = crop.resize((144, 48), Image.BILINEAR)

            new_img_path = f'{i}.jpg'
            crop.save(os.path.join(img_dir, new_img_path))
            f.write(f'{new_img_path}\t{label}\n')


def create_ccpd_dataset(ccpd_dir, output_dir, train_split=0.5):
    train_img_dir = os.path.join(output_dir, 'train', 'images')
    val_img_dir = os.path.join(output_dir, 'val', 'images')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)

    train_label_path = os.path.join(output_dir, 'train', 'labels.txt')
    val_label_path = os.path.join(output_dir, 'val', 'labels.txt')

    images = [f for f in os.listdir(ccpd_dir) if f.endswith('.jpg')]
    random.shuffle(images)

    split = int(len(images) * train_split)
    train_images = images[:split]
    val_images = images[split:]

    print("## training set preparation ##")
    create_subfolder(train_images, train_img_dir, train_label_path)
    print("## validation set preparation ##")
    create_subfolder(val_images, val_img_dir, val_label_path)



if __name__ == '__main__':
    ccpd_dir = 'C:/Users/User/Desktop/progetto/CCPD2019/ccpd_base' 
    output_dir = 'C:/Users/User/Desktop/progetto/dati/Database_split'

    create_ccpd_dataset(ccpd_dir, output_dir, train_split=0.8)
