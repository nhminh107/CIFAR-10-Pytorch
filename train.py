from torch.optim import AdamW
import CONFIG as cf 
from data import ImageDataset
from model import CNN_Model
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])
img_dataset = ImageDataset(transform=train_transform)
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

best_valid_loss = 100
for epoch in range(40):
    cnn_model.train()
    print(f"Epoch: {epoch}")

    print(f"Training Epoch {epoch}")
    training_loss = 0.0 
    training_loss_avg = 0.0 
    train_accuracy = 0

    total_img = 0
    correct_img = 0 
    for i, data in enumerate(train_imgs, 0):
        total_img += cf.batch_size
        inputs, labels = data
        inputs = inputs.to(cf.device)
        labels = labels.to(cf.device)
        cnn_model.zero_grad()
        outputs = cnn_model(inputs)
        predictions = outputs.argmax(dim=1)
        correct_img += (predictions == labels).sum().item()
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
        training_loss_avg += loss.item()*cf.batch_size
        if (i + 1) % 100 == 0:
            print(
                f"Batch {i + 1}: "
                f"loss = {training_loss / 100:.4f}"
            )
        training_loss = 0.0 
    train_accuracy = 1.0*correct_img/total_img
    training_loss_avg /= total_img
    print(f"Validation Epoch {epoch}")
    valid_loss = 0.0 
    valid_accuracy = 0.0

    total_valid_img = 0
    correct_valid_img = 0 

    cnn_model.eval()

    with torch.inference_mode():
        for i, data in enumerate(valid_imgs):
            total_valid_img += cf.batch_size
            imgs, labels = data
            imgs = imgs.to(cf.device)
            labels = labels.to(cf.device)
            outputs = cnn_model(imgs)
            predictions = outputs.argmax(dim=1)
            correct_valid_img += (predictions == labels).sum().item()
            loss = loss_func(outputs, labels)
            valid_loss += loss.item()*cf.batch_size

    valid_accuracy = 1.0*correct_valid_img/total_valid_img
    print(f"Train accuracy {train_accuracy}")
    print(f"Valid accuracy {valid_accuracy}")
    print(f"Train loss {training_loss_avg}")
    print(f"Valid loss {valid_loss/total_valid_img}")

    if (valid_loss/total_valid_img) < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(cnn_model.state_dict(), "best_model.pth")
        print(f"Save model as epoch {epoch}")

    print("---------------------")




