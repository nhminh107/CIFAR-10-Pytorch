import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import CONFIG as cf
from data import ImageDataset
from model import CNN_Model


TEST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])


def load_model(checkpoint_path, device):
    model = CNN_Model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict(model, data_loader, device):
    class_names = {class_id: name for name, class_id in cf.class_dict.items()}
    image_ids = []
    labels = []

    with torch.inference_mode():
        for images, batch_ids in data_loader:
            images = images.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1).cpu().tolist()

            image_ids.extend(batch_ids.tolist())
            labels.extend(class_names[class_id] for class_id in predictions)

    return pd.DataFrame({"id": image_ids, "label": labels})


def parse_args():
    parser = argparse.ArgumentParser(description="Predict CIFAR-10 test labels")
    parser.add_argument("--checkpoint", default="best_model.pth")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--batch-size", type=int, default=cf.batch_size)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    test_dataset = ImageDataset(
        split="test",
        data_dir=args.data_dir,
        transform=TEST_TRANSFORM,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=cf.device.type == "cuda",
    )

    model = load_model(checkpoint_path, cf.device)
    submission = predict(model, test_loader, cf.device)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
