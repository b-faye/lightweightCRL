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
# Raplace addition by other fusion methods
from addition.model import Model
from configs import CFG
from PIL import Image
import urllib.request
import zipfile
import torch.nn as nn


def download_and_unzip_tiny_imagenet(download_dir='../../datasets/tiny_imagenet'):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    # Download TinyImageNet dataset
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    zip_file_path = os.path.join(download_dir, 'tiny_imagenet.zip')
    urllib.request.urlretrieve(url, zip_file_path)

    # Extract the downloaded zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)

    # Remove the zip file
    os.remove(zip_file_path)

    print("TinyImageNet dataset downloaded and unzipped successfully!")


class CustomTinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.classes = os.listdir(os.path.join(root_dir, 'train'))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        if self.train:
            self.data_dir = os.path.join(root_dir, 'train')
            self.annotations_file = os.path.join(root_dir, 'wnids.txt')
            with open(self.annotations_file, 'r') as f:
                self.classes = sorted([line.strip() for line in f.readlines()])
            self.image_paths = []
            for cls in self.classes:
                images_dir = os.path.join(self.data_dir, cls, 'images')
                for img_name in os.listdir(images_dir):
                    self.image_paths.append((os.path.join(images_dir, img_name), self.class_to_idx[cls]))
        else:
            self.data_dir = os.path.join(root_dir, 'val')
            self.annotations_file = os.path.join(root_dir, 'val', 'val_annotations.txt')
            self.image_paths = []
            with open(self.annotations_file, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    img_cls = parts[1]
                    img_cls_idx = self.class_to_idx[img_cls]
                    img_path = os.path.join(self.data_dir, 'images', img_name)
                    self.image_paths.append((img_path, img_cls_idx))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label


class Classifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=200, hidden_dim=256, device='cpu', *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.model = pretrained_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fusion_encoder.projection_dim, hidden_dim),
            nn.Linear(hidden_dim, self.num_classes)
        )

        self.to(self.device)
        self.model.to(self.device)
        self.classifier.to(self.device)

    def forward(self, inputs):
        image_features = self.model.encode_image(image_tensors=inputs)
        logits = self.classifier(image_features)
        return logits

    def accuracy(self, data_loader):
        top_accuracy = {f"top_{i+1}_accuracy": 0 for i in range(5)}
        total_samples = 0
        with torch.no_grad():
            self.eval()
            for inputs, labels in tqdm(data_loader, desc='Validation'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                total_samples += labels.size(0)
                logits = self(inputs)
                _, predicted_top_k = torch.topk(logits, 5, dim=1)
                for i in range(5):
                    top_accuracy[f"top_{i+1}_accuracy"] += torch.sum(torch.any(
                        predicted_top_k[:, :i+1] == labels.view(-1, 1), dim=-1)).item()

        for name in top_accuracy:
            top_accuracy[name] /= total_samples

        return top_accuracy

    def __call__(self, inputs):
        return self.forward(inputs)


def main(model_path="../../weights/model.pt", epochs=100, device=CFG.device,
         root_dir="../../datasets/tiny_imagenet/tiny-imagenet-200", download_dataset=False):
    if download_dataset:
        download_and_unzip_tiny_imagenet()
    pretrained_model = Model(device=device)
    pretrained_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    train_data = CustomTinyImageNetDataset(root_dir=root_dir, transform=pretrained_model.image_preprocessor, train=True)
    test_data = CustomTinyImageNetDataset(root_dir=root_dir, transform=pretrained_model.image_preprocessor, train=False)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    model = Classifier(pretrained_model=pretrained_model, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        if (epoch + 1) % 5 == 0:
            # Validate the model on the test set
            top_accuracy = model.accuracy(test_loader)
            print(top_accuracy)

    # Save the fine-tuned model if needed
    torch.save(model.state_dict(), 'fine_tuned_model.pt')


if __name__ == "__main__":
    main()
