import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import CONFIG as cf
from data import ImageDataset
from model import CNN_Model


# Test set chỉ cần chuẩn hóa giống lúc validation, không dùng augmentation.
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

test_dataset = ImageDataset(
    split="test",
    data_dir="data",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=cf.batch_size,
    shuffle=False
)

# Khởi tạo lại model rồi nạp trọng số tốt nhất đã lưu khi train.
model = CNN_Model().to(cf.device)
model.load_state_dict(
    torch.load("best_model.pth", map_location=cf.device, weights_only=True)
)
model.eval()

ids = []
labels = []
id_to_class = {class_id: name for name, class_id in cf.class_dict.items()}

with torch.inference_mode():
    for images, image_ids in test_loader:
        images = images.to(cf.device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        ids.extend(image_ids.tolist())
        labels.extend(id_to_class[prediction] for prediction in predictions)

submission = pd.DataFrame({
    "id": ids,
    "label": labels
})

submission.to_csv("submission.csv", index=False)
print("Đã lưu kết quả vào submission.csv")
