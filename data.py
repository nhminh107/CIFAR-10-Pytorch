import cv2
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import CONFIG as cf
import os


class ImageDataset(Dataset):
    def __init__(self, transform = None, target_transform=None):
        self.img_labels = pd.read_csv('data/trainLabels.csv')
        self.img_dir = 'data/train'
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
        # CIFAR has 50000 images, but I will split it to train and valid
        # TRAIN :40000 - VALID : 10000
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[index, 0]) + ".png")
        image = Image.open(img_path)
        label = self.img_labels.iloc[index, 1]

        if self.transform: 
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label_int = cf.class_dict[label]
        return image, label_int
    

if __name__ == "__main__":  

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_d = ImageDataset()
    test1 = img_d.__getitem__(10)
    img = test1[0]
    img.show()
        

