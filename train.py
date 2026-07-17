from torch.optim import AdamW
import CONFIG as cf 
from data import ImageDataset
from model import CNN_Model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])
img_dataset = ImageDataset(transform=transform)
cnn_model = CNN_Model().to(cf.device)
optimizer = AdamW(
    cnn_model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
loss_func = nn.CrossEntropyLoss()


train_dataset, valid_dataset = torch.utils.data.random_split(img_dataset, [0.8, 0.2])
train_imgs = DataLoader(train_dataset, batch_size=cf.batch_size, pin_memory=True, shuffle=True)
valid_imgs = DataLoader(valid_dataset, batch_size=cf.batch_size, pin_memory=True, shuffle=True)

for epoch in range(10):
    cnn_model.train()
    print(epoch)

    training_loss = 0.0 
    for i, data in enumerate(train_imgs, 0):

        inputs, labels = data
        inputs = inputs.to(cf.device)
        labels = labels.to(cf.device)
        cnn_model.zero_grad()
        outputs = cnn_model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        
        if (i + 1) % 100 == 0:
            print(
                f"Batch {i + 1}: "
                f"loss = {training_loss / 100:.4f}"
            )
        training_loss = 0 


