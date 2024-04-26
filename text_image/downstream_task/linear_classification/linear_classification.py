import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
root_dir = "/".join(root_dir.split("/")[:-1])
sys.path.append(root_dir)
from tqdm import tqdm
import numpy as np
import torch
# Raplace addition by other fusion methods
from addition.model import Model
from configs import CFG
import torch.nn as nn
import torchvision


class CustomCIFAR100Dataset(torch.utils.data.Dataset):
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


class LinearClassifier(nn.Module):
    def __init__(self, pretrained_model, num_classes=100, hidden_dim=128, trainable=False, device='cpu', *args, **kwargs):
        super(LinearClassifier, self).__init__(*args, **kwargs)
        self.model = pretrained_model
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.trainable = trainable
        self.device = device
        self.classifier = nn.Sequential(
            nn.Linear(self.model.fusion_encoder.projection_dim, hidden_dim),
            nn.Linear(hidden_dim, self.num_classes)
        )
        for parameter in self.model.parameters():
            parameter.requires_grad = self.trainable
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


def main(model_path="../../weights/model.pt", trainable=False, epochs=100, device=CFG.device):
    pretrained_model = Model(device=device)
    pretrained_model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    train_data = torchvision.datasets.CIFAR100(root='./data', train=True, download=True)
    train_data = CustomCIFAR100Dataset(train_data, transform=pretrained_model.image_preprocessor)
    test_data = torchvision.datasets.CIFAR100(root='./data', train=False, download=True)
    test_data = CustomCIFAR100Dataset(test_data, transform=pretrained_model.image_preprocessor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
    model = LinearClassifier(pretrained_model=pretrained_model, device=device, trainable=trainable)
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
