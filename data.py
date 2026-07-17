import cv2
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import CONFIG as cf
import os


class ImageDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, split="train", data_dir="data"):
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        self.split = split
        self.img_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.target_transform = target_transform

        if split == "train":
            self.img_labels = pd.read_csv(os.path.join(data_dir, "trainLabels.csv"))
        else:
            # Kaggle's CIFAR-10 test images are named <id>.png. Sorting by the
            # numeric id makes both inference and the generated CSV deterministic.
            self.image_ids = sorted(
                int(os.path.splitext(filename)[0])
                for filename in os.listdir(self.img_dir)
                if filename.lower().endswith(".png")
            )

    def __len__(self):
        if self.split == "train":
            return len(self.img_labels)
        return len(self.image_ids)
        # CIFAR has 50000 images, but I will split it to train and valid
        # TRAIN :40000 - VALID : 10000
    
    def __getitem__(self, index):
        if self.split == "train":
            image_id = int(self.img_labels.iloc[index, 0])
            label = self.img_labels.iloc[index, 1]
        else:
            image_id = self.image_ids[index]

        img_path = os.path.join(self.img_dir, f"{image_id}.png")
        with Image.open(img_path) as source_image:
            image = source_image.convert("RGB")

        if self.transform: 
            image = self.transform(image)

        if self.split == "test":
            return image, image_id

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
        
