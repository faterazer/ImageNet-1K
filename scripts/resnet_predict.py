import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data import TestDataset

batch_size = 200
device = torch.device("cuda")


if __name__ == "__main__":
    test_dataset = TestDataset("/mnt/bn/bytenas0/ImageNet-1K/data/test", transforms=ResNet50_Weights.IMAGENET1K_V2.transforms())
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().cuda()

    preds = []
    for X in tqdm(test_dataloader):
        assert isinstance(X, torch.Tensor)
        X = X.to(device)
        with torch.no_grad():
            probs = model(X)
        _, idxs = torch.topk(probs, k=5)
        preds.append(idxs.cpu().numpy() + 1)
    preds = np.concatenate(preds, axis=0).tolist()

    with open("./submission.txt", "w") as fp:
        for p in preds:
            fp.write(" ".join(map(str, p)) + "\n")
