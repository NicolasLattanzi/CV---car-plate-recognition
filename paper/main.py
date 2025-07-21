import torch
from torchvision import transforms
from PIL import Image
from pdlpr_git import PDLPR
import my_utils

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tran=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) ])
#carico primo modello
yolo=torch.hub.load('ultralytics/yolov5', 'custom', path="E:/progetto edoardo CV/CV---car-plate-recognition-main/paper/yolov5/runs/train/best.pt")  ##path da aggiornare
yolo.conf=0.4
yolo.to(device)
yolo.eval()
#carico secondo modello
model = PDLPR(
    in_channels=3,             # o il valore/numero di canali delle tue immagini
    base_channels=256,         # o quello che preferisci
    encoder_d_model=256,
    encoder_nhead=4,
    encoder_height=16,
    encoder_width=16,
    decoder_num_layers=2,
    num_classes=69,            # aggiorna se hai un numero diverso di classi possibili!
    seq_len=7                  # aggiorna se la lunghezza della sequenza è diversa
).to(device)

model.load_state_dict(torch.load("E:/progetto edoardo CV/CV---car-plate-recognition-main/paper/pdlpr_final.pth"))
model.eval()

image_path='E:/progetto edoardo CV/CV---car-plate-recognition-main/paper/prova.jpg'
#pipeline!

img=Image.open(image_path).convert('RGB')   #apre la foto

output1=yolo(img)   #da la foto a yolo che ritorno il box della targa
border=output1.pandas().xyxy[0] #prende i dati della box e li salva per il crop
if len(border)==0:
    print("targa non rilevata")
    
else:
    bb=border.iloc[0]  #crea la box con i dati con i dati ottenuti da yolo
    a1,a2,b1,b2= map(int, [bb['xmin'], bb['ymin'] , bb['xmax'] , bb['ymax'] ]) 
    cropImg=img.crop((a1,a2,b1,b2))  #crop della foto
    cropImgTens=tran(cropImg).unsqueeze(0).to(device) #trasformazione in tenrosflow per darla al secondo modello
    with torch.no_grad():
        output2=model(cropImgTens)   #passaggio al secondo modello
        predict=output2.argmax(dim=-1).squeeze(0).cpu().numpy()  #prende i caratteri piu probabili ottenuti dall'output del secondo modello

    decodedPlate=my_utils.lpDecoder(predict)
    print("targa decodificata: ",decodedPlate)



