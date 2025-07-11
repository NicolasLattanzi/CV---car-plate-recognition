from torchvision.datasets import ImageFolder    
from torchvision import transforms

import utils


class MyImageFolder(ImageFolder):

    def __getitem__(self, index: int) -> tuple[any, any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path
    

def create_dataset(path):

    transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    return MyImageFolder(root=path, transform=transform)
