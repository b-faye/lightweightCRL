import os
import requests
import zipfile
import torch
from pycocotools.coco import COCO
import albumentations as A
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from configs import CFG


def download_dataset(data_dir="../datasets"):
    # Create caption and image directories
    annotations_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "train2014")

    # Download annotations (captions)
    zip_file = os.path.join(annotations_dir, "annotations.zip")
    url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    response = requests.get(url, stream=True)
    # write chunk in zip file
    with open(zip_file, "wb") as f:
        # 8192 = 8KB chunks (block or piece of data)
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)  # Extract all contents to the specified directory
    os.remove(zip_file)

    # Download images
    zip_file = os.path.join(images_dir, "train2014.zip")
    url = "http://images.cocodataset.org/zips/train2014.zip"
    response = requests.get(url, stream=True)

    # write chunk in zip file
    with open(zip_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    # unzip file
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(data_dir)  # Extract all contents to the specified directory
    os.remove(zip_file)


def make_positive_pairs(annotation_json_file, model=SentenceTransformer('all-MiniLM-L6-v2'), threshold=0.7):
    coco = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    positive_pairs = []
    for image_id in tqdm(image_ids, desc="Pairs"):
        # annotation_ids: list of caption ids to image_id []
        annotation_ids = coco.getAnnIds(image_id)
        # annotations: list of dictionary, each dictionary for one caption [{}, {}, ..., {}]
        annotations = coco.loadAnns(annotation_ids)

        # Pair captions from the same image
        for i in range(len(annotations)):
            if 'caption' in annotations[i]:
                caption1 = annotations[i]['caption']
                positive_pairs.append((caption1, caption1))
                embeddings1 = model.encode(caption1, convert_to_tensor=True)
                for j in range(i+1, len(annotations)):
                    if 'caption' in annotations[j]:
                        caption2 = annotations[j]['caption']
                        # Compute embedding for both lists
                        embeddings2 = model.encode(caption2, convert_to_tensor=True)
                        if util.cos_sim(embeddings1, embeddings2) >= threshold:
                            positive_pairs.append((caption1, caption2))
                            positive_pairs.append((caption2, caption1))

    return positive_pairs


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, texts, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        self.texts = texts

    def __getitem__(self, index):

        return self.texts[index][0], self.texts[index][1]

    def __len__(self):
        return len(self.texts)


def build_loaders(dataframe):
    dataset = DataGenerator(dataframe)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
    )
    return dataloader
