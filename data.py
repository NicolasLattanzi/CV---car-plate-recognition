from torchvision.datasets import ImageFolder    
from torchvision import transforms


def create_dataset(path):

    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return ImageFolder(root=path, transform=transform)
