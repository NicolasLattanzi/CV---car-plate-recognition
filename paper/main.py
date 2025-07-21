import torch
from torchvision import transforms
from PIL import Image
#sys.path.append("")
from paper import pdlpr_git
import utils

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tran=transforms.Compose([transforms.Resize((128,128)),transforms.ToTensor(), transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) ])
#carico primo modello
yolo=torch.hub.load('ultralytics/yolov5', 'custom', path='inserisci path qui dopo che il train è finito.')  ##path da aggiornare
yolo.conf=0.4
yolo.to(device)
yolo.eval()
#carico secondo modello
model=pdlpr_git(id=3,base=256,encoderD=256, encoderNh=8, encoderH=16, encoderW=16, decoderLay=2, nClass=69, seqL=7).to(device)

model.load_state_dict(torch.load("E:\progetto edoardo CV\CV---car-plate-recognition-main\paper\pdlpr_final.pth"))
model.eval()

image_path=''
#pipeline!

img=Image.open(image_path).convert('RGB')   #apre la foto

output1=yolo(img)   #da la foto a yolo che ritorno il box della targa
border=output1.pandas().xyxy[0] #prende i dati della box e li salva per il crop
if len(border)==0:
    print("targa non rilevata")
    
else:
    bb=border.iloc[0]  #crea la box con i dati con i dati ottenuti da yolo
    a1,a2,b1,b2= map(int, [bb['xmin'], bb['ymin'] , bb['xmax'] , bb['ymaz'] ]) 
    cropImg=img.crop((a1,a2,b1,b2))  #crop della foto
    cropImgTens=tran(cropImg).unsqueeze(0).to(device) #trasformazione in tenrosflow per darla al secondo modello
    with torch.no_grad():
        output2=model(cropImgTens)   #passaggio al secondo modello
        predict=output2.argmax(dim=-1).squeeze(0).numpy  #prende i caratteri piu probabili ottenuti dall'output del secondo modello

    decodedPlate=utils.lpDecoder(predict)
    print("targa decodificata: ",decodedPlate)



