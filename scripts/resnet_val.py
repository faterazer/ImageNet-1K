import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import top_k_accuracy_score
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

batch_size = 100
device = torch.device("cuda")
preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()


def transform(examples: dict) -> dict:
    return {
        "data": [preprocess(image.convert("RGB")) for image in examples["image"]],
        "labels": torch.LongTensor(examples["label"]),
    }


val_dataset = load_dataset("imagenet-1k", split="validation")
val_dataset.set_transform(transform)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().cuda()

y_score, y_true = [], []
for batch in tqdm(val_dataloader):
    X, y = batch["data"].to(device), batch["labels"]
    with torch.no_grad():
        X_hat = model(X)
    y_score.append(X_hat.cpu().numpy())
    y_true.append(y)
y_score = np.concatenate(y_score, axis=0)
y_true = np.concatenate(y_true)
print("Top 1 Acc:", top_k_accuracy_score(y_true=y_true, y_score=y_score, k=1))
print("Top 5 Acc:", top_k_accuracy_score(y_true=y_true, y_score=y_score, k=5))
