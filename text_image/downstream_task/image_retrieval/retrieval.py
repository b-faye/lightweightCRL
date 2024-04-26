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
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = self.load_annotations(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_filename, caption = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, img_filename)

        # Load the image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(image=img)['image']

        return img, caption

    @classmethod
    def load_annotations(cls, annotation_file):
        # Load annotations from the CSV file into a DataFrame
        df = pd.read_csv(annotation_file)

        # Optionally, you can preprocess the DataFrame if needed
        # For example, you can merge duplicate rows or perform other data cleaning

        return df


# Load and unzip flickr8K on datasets directory
def get_embeddings(model, root_dir="../../datasets/flickr8K/Images",
                   annotation_file="../../datasets/flickr8K/captions.txt",  batch_size=32):

    custom_dataset = CustomDataset(root_dir, annotation_file, transform=model.image_preprocessor)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=False)
    image_embeddings = []
    caption_embeddings = []
    model.eval()
    with torch.no_grad():
        for image_tensors, captions in tqdm(data_loader, desc="Validation"):
            image_tensors = image_tensors.to(model.device)  # Move image tensors to device
            image_embeddings.append(model.encode_image(image_tensors=image_tensors))
            caption_embeddings.append(model.encode_text(captions))

    image_embeddings = torch.cat(image_embeddings)
    caption_embeddings = torch.cat(caption_embeddings)

    torch.save(image_embeddings, "image_embeddings.pt")
    torch.save(caption_embeddings, "caption_embeddings.pt")
    return image_embeddings, caption_embeddings, custom_dataset


# Image retrieval accuracy

def image_retrieval_accuracy(image_embeddings, text_embeddings, batch_size=100, temperature=2.5, device='cpu'):
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    accuracy = {}
    labels = torch.arange(start=0, end=text_embeddings.size(0))

    for start_idx in tqdm(range(0, text_embeddings.size(0), batch_size), desc='Accuracy'):
        end_idx = min(start_idx + batch_size, text_embeddings.size(0))
        text_batch = text_embeddings[start_idx:end_idx]

        dot_similarities = (text_batch @ image_embeddings.T) * torch.exp(torch.tensor(temperature).to(device))

        _, top_k = dot_similarities.cpu().topk(10100, dim=-1)

        for i in range(100, 10100, 100):
            # top_1_accuracy is the best
            accuracy[f"top_{i}_accuracy"] = (
                    torch.sum(torch.any(top_k[:, :i] == labels[start_idx:end_idx].view(-1, 1),
                                        dim=-1)).item() / text_batch.size(0))

    return accuracy


# Text retrieval accuracy

def text_retrieval_accuracy(image_embeddings, text_embeddings, dataframe, batch_size=100,
                            temperature=2.5, device='cpu'):
    num_groups = dataframe.shape[0] // 5
    labels = torch.tensor([[i + j for j in range(5)] for i in range(0, num_groups*5, 5)])
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    accuracy = {}

    for start_idx in tqdm(range(0, image_embeddings.size(0), batch_size), desc='Accuracy'):
        end_idx = min(start_idx + batch_size, image_embeddings.size(0))
        image_batch = image_embeddings[start_idx:end_idx]
        dot_similarities = (image_batch @ text_embeddings.T) * torch.exp(torch.tensor(temperature).to(device))

        _, top_k = dot_similarities.cpu().topk(5, dim=-1)

        for i in range(5):
            results = []
            for row_a, row_b in zip(top_k, labels[start_idx:end_idx]):
                set_a = set(row_a.tolist())
                set_b = set(row_b.tolist())
                # If in the retrieved texts (topk) at least i+1 texts are in the correct captions (5)
                result = len(set_a.intersection(set_b)) >= i+1
                results.append(result)
            # top_5_accuracy is the best (all right captions are selected)
            accuracy[f"top_{i+1}_accuracy"] = (
                sum(results) / image_batch.size(0))

    return accuracy


def main(model_path="../../weights/model.pt", device=CFG.device):
    model = Model(device=device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    image_embeddings, caption_embeddings, custom_dataset = get_embeddings(model=model)
    accuracy = image_retrieval_accuracy(image_embeddings, caption_embeddings, device=model.device)
    print(f"Image Retrieval: {accuracy}")
    accuracy = text_retrieval_accuracy(image_embeddings, caption_embeddings,
                                       custom_dataset.annotations, device=model.device)
    print(f"Text Retrieval: {accuracy}")


if __name__ == "__main__":
    main()
