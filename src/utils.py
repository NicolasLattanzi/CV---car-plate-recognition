import torch
from torchvision import transforms
from torchvision.transforms import functional
from globals import provinces, alphabets, ads


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
def rename_files(folder_path):
    """
    Rinomina tutti i file della cartella specificata mantenendo solo la quinta parte
    del nome originale, separata da trattini ('-').
    """
    seen=set()
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Salta se non è un file
        if not os.path.isfile(file_path):
            continue

        name, ext = os.path.splitext(filename)
        parts = name.split('-')
        # Assicurati che il nome ha almeno 5 segmenti
        if len(parts) > 4:
            new_name = parts[4] + ext
            new_path = os.path.join(folder_path, new_name)
            if new_name in seen or os.path.exists(new_path):
                os.remove(file_path)
                print(f'Eliminato duplicato: "{filename}"')
            else:
                os.rename(file_path, new_path)
                seen.add(new_name)
                print(f'Rinominato: "{filename}" -> "{new_name}"')
        else:
            print(f'Saltato: "{filename}" (struttura non valida)')
def lpDecoder(license,pr_list=provinces, char_list=ads):
    
    decoded=[]
    print(license)
    for idx in license.squeeze():
        if(len(decoded)==0):
            decoded.append(pr_list[idx])
        else:
            decoded.append(char_list[idx])
    return ''.join(decoded)
    
