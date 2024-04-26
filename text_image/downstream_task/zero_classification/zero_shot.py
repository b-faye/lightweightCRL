import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
root_dir = "/".join(root_dir.split("/")[:-1])
sys.path.append(root_dir)
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from addition.model import Model
from configs import CFG
import torchvision
from torch.utils.data import DataLoader


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        # Convert PIL Image to NumPy array
        img = np.array(img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']

        return img, label


def zero_shot_classification(model, dataloader, unique_class_names, temperature=2.5):

    accuracy = {f"top_{i+1}_accuracy": 0 for i in range(5)}
    # Encode label names
    text_features = model.encode_text(texts=[f"a photo of a {name}" for name in unique_class_names])
    text_features = F.normalize(text_features, p=2, dim=-1)
    model.eval()
    with torch.no_grad():
        for id, batch in enumerate(tqdm(dataloader, desc="Testing")):
            image_tensors, labels = batch
            # Encode images
            image_features = model.encode_image(image_tensors=image_tensors)
            image_features = F.normalize(image_features, p=2, dim=-1)
            similarities = (image_features @ text_features.T) * torch.exp(torch.tensor(temperature).to(model.device))
            text_probs = (10.0 * similarities).softmax(dim=-1)
            _, top_k = text_probs.cpu().topk(5, dim=-1)
            for i in range(5):
                accuracy[f"top_{i+1}_accuracy"] += torch.sum(
                    torch.any(top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

    return accuracy


def main(model_path="../../weights/model.pt"):
    model = Model(device=CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    class_names = test_set.classes
    test_set = CustomCIFAR10Dataset(test_set, transform=model.image_preprocessor)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    accuracy = zero_shot_classification(model, test_loader, class_names)
    for name, score in accuracy.items():
        print(f"{name}: {score / test_set.dataset.data.shape[0]}")


if __name__ == "__main__":
    main()
