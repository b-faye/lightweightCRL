import os
import requests
import zipfile
import torch
from pycocotools.coco import COCO
import albumentations as A
import cv2
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


def make_pairs(annotation_json_file, image_dir, max_captions=3):
    coco = COCO(annotation_json_file)
    image_ids = list(coco.imgs.keys())

    image_caption = []
    for image_id in image_ids:
        num_caption = 0
        # image_info: list of one dict [{}]
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info["file_name"])
        # annotation_ids: list of caption ids to image_id []
        annotation_ids = coco.getAnnIds(image_id)
        # annotations: list of dictionary, each dictionary for one caption [{}, {}, ..., {}]
        annotations = coco.loadAnns(annotation_ids)
        for annotation in annotations:
            if 'caption' in list(annotation.keys()) and num_caption < max_captions:
                caption = annotation['caption']
                image_caption.append((image_path, caption))
                num_caption += 1

    return image_caption


class DataGenerator(torch.utils.data.Dataset):
    def __init__(self, images, captions, tokenizer, transformer, *args, **kwargs):
        super(DataGenerator, self).__init__(*args, **kwargs)
        # Image processor
        self.transformer = transformer
        # List of images
        self.images = images
        # List of captions
        self.captions = captions
        # Tokenize all captions
        # Return: dict of keys "input_ids", "token_type_ids", and "attention_mask"
        # each value is list of lists (number of captions): [[], [], ..., []]
        self.tokenized_captions = tokenizer(
            list(self.captions), padding=True, truncation=True, max_length=CFG.max_length
        )

    def __getitem__(self, index):
        # Select caption in position index
        item = {
            k: torch.tensor(v[index]) for k, v in self.tokenized_captions.items()
        }
        # Select image in position index
        image = cv2.imread(f"{self.images[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transformer(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        # Caption text
        item["caption"] = self.captions[index]
        return item

    def __len__(self):
        return len(self.captions)


# Resize and Normalize image
def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
            ]
        )


def build_loaders(dataframe, tokenizer, mode):
    transformer = get_transforms(mode=mode)
    dataset = DataGenerator(
        [img for img, _ in dataframe],
        [caption for _, caption in dataframe],
        tokenizer=tokenizer,
        transformer=transformer,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader
