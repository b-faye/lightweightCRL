import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
from baseline import ImageEncoder, TextEncoder
from fusion import Fusion, ContextEncoder
from transformers import BertTokenizer
from albumentations.pytorch import ToTensorV2
from configs import CFG
import albumentations as A
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import requests
import cv2
from tqdm import tqdm
from coco_dataset import build_loaders
from loss import AvgMeter


class Model(nn.Module):

    def __init__(self, image_encoder=ImageEncoder(), text_encoder=TextEncoder(), context_encoder=ContextEncoder(),
                 fusion_encoder=Fusion(), device='cpu', tokenizer=BertTokenizer.from_pretrained(CFG.bert_name),
                 image_preprocessor=A.Compose([A.Resize(CFG.image_size, CFG.image_size, always_apply=True),
                 A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True), ToTensorV2()]),
                 *args, **kwargs):
        """
        Initialize the model.

        :param image_encoder: Image encoder module (default: ImageEncoder()).
        :param text_encoder: Text encoder module (default: TextEncoder()).
        :param context_encoder: Context encoder module (default: ContextEncoder()).
        :param fusion_encoder: Fusion encoder module (default: Fusion()).
        :param device: Device to run the model on (default: 'cpu').
        :param tokenizer: Tokenizer for text encoding (default: BertTokenizer.from_pretrained(CFG.bert_name)).
        :param image_preprocessor: Preprocessor for image inputs (default: A.Compose([...])).
        """

        super(Model, self).__init__(*args, **kwargs)
        self.device = device
        self.to(self.device)
        self.image_encoder = image_encoder.to(self.device)
        self.text_encoder = text_encoder.to(self.device)
        self.context_encoder = context_encoder.to(self.device)
        self.context_encoder.device = self.device
        self.fusion_encoder = fusion_encoder.to(self.device)
        self.tokenizer = tokenizer
        self.image_preprocessor = image_preprocessor
        self.image_context_id = 1
        self.text_context_id = 0
        # The learnable temperature parameter τ was initialized to the equivalent of 0.07 from (Wu et al., 2018)
        # and clipped to prevent scaling the logits by more than 100, which we found necessary
        # to prevent training instability.
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))

    @classmethod
    def load_image(cls, image_path):
        # Load online image
        if image_path.startswith("http"):
            response = requests.get(image_path)
            # Check if the request was successful
            if response.status_code == 200:
                # Convert the image content to a numpy array
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)

                # Decode the image using OpenCV
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load local image
        else:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def encode_image(self, image_paths=None, image_tensors=None, outputs="average"):
        """
        Encode images into feature vectors.

        :param image_paths: List of image paths.
        :param image_tensors: Torch tensor (batch, 3, 224, 224).
        :param outputs type of outputs: average, min, max or sequence
        :return: Encoded image features.
        """
        if image_paths is not None:
            image_processed = [self.image_preprocessor(image=self.load_image(image))["image"] for image in image_paths]
            image_processed = torch.stack(image_processed).to(self.device)
            with torch.no_grad():
                image_features = self.image_encoder(image_processed.to(self.device))
                image_context_feature = self.context_encoder(torch.tensor([self.image_context_id],
                                                                    dtype=torch.float32).unsqueeze(1).to(self.device))
                image_context_features = image_context_feature.repeat(image_processed.size(0), 1)
                output_features = self.fusion_encoder([image_features, image_context_features])

        elif image_tensors is not None:

            with torch.no_grad():
                image_features = self.image_encoder(image_tensors.to(self.device))
                image_context_feature = self.context_encoder(
                    torch.tensor([self.image_context_id], dtype=torch.float32).unsqueeze(1).to(self.device))
                image_context_features = image_context_feature.repeat(image_tensors.size(0), 1)
                output_features = self.fusion_encoder([image_features, image_context_features])
        if outputs == "average":
            image_features = output_features.average_outputs
        elif outputs == "min":
            image_features = output_features.min_outputs
        elif outputs == "max":
            image_features = output_features.max_outputs
        else:
            image_features = output_features.sequence_outputs

        return image_features

    def encode_text(self, texts, max_length=128, outputs="average"):
        """
        Encode text descriptions into feature vectors.

        :param texts: List of text descriptions.
        :param max_length: Maximum length of the text sequences (default: 128).
        :param outputs type of outputs: average, min, max or sequence
        :return: Encoded text features.
        """
        encoded_query = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length
        )
        batch = {
            key: torch.tensor(values).to(self.device)
            for key, values in encoded_query.items()
        }
        with torch.no_grad():
            text_features = self.text_encoder([
                batch["input_ids"], batch["attention_mask"]
            ])
            text_context_feature = self.context_encoder(
                torch.tensor([self.text_context_id], dtype=torch.float32).unsqueeze(1).to(self.device))
            text_context_features = text_context_feature.repeat(len(texts), 1)
            output_features = self.fusion_encoder([text_features, text_context_features])
            if outputs == "average":
                text_features = output_features.average_outputs
            elif outputs == "min":
                text_features = output_features.min_outputs
            elif outputs == "max":
                text_features = output_features.max_outputs
            else:
                text_features = output_features.sequence_outputs
        return text_features

    def matching(self, image_paths, texts, normalize=True, top_k=None, strategy="similarity", temperature=1.0):
        """
        Calculate similarities between images and texts.

        :param image_paths: List of paths to images.
        :param texts: List of text descriptions.
        :param normalize: Whether to normalize the features (default: True).
        :param top_k: Return top K results (default: None).
        :param strategy: Matching strategy, either 'similarity' or 'softmax' (default: 'similarity').
        :param temperature: change real distribution, default = 2.5
        :return: If top_k is provided, returns top probabilities and labels, otherwise returns dot similarities.
        """
        image_features = self.encode_image(image_paths=image_paths)
        text_features = self.encode_text(texts=texts)

        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
            text_features = F.normalize(text_features, p=2, dim=-1)
        dot_similarities = (image_features @ text_features.T) * torch.exp(torch.tensor(temperature).to(self.device))
        if strategy == 'softmax':
            dot_similarities = (float(len(set(texts))) * dot_similarities).softmax(dim=-1)
        if top_k is not None:
            top_probs, top_labels = dot_similarities.cpu().topk(top_k, dim=-1)
            return top_probs, top_labels
        else:
            return dot_similarities, None

    def image_retrieval(self, query, image_paths, image_embeddings=None, temperature=1.0, n=9, plot=False):
        """
        Perform image retrieval based on a text query.

        :param query: Text query (string).
        :param image_paths: List of image paths (optional).
        :param image_embeddings: Precomputed image embeddings (optional).
        :param temperature: change real distribution, default = 2.5
        :param n: Number of images to retrieve (default: 9).
        :param plot: Whether to plot the retrieved images (default: False).
        :return: Tuple containing similarity values and indices of the retrieved images.
        """
        text_embeddings = self.encode_text([query])
        if image_embeddings is None:
            image_embeddings = self.encode_image(image_paths=image_paths)

        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = (text_embeddings_n @ image_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))
        if n > len(image_paths):
            n = len(image_paths)
        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        if plot:
            nrows = int(np.sqrt(n))
            ncols = int(np.ceil(n / nrows))
            matches = [image_paths[idx] for idx in indices]
            fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
            for match, ax in zip(matches, axes.flatten()):
                image = self.load_image(f"{match}")
                ax.imshow(image)
                ax.axis("off")

            fig.suptitle(query)
            plt.show()
        return values, indices

    def text_retrieval(self, query, texts, text_embeddings=None, n=9, plot_image=False, temperature=1.0):
        """
        Perform text retrieval based on an image query.

        :param query: Image query (path of image).
        :param texts: List of text samples.
        :param text_embeddings: Precomputed text embeddings (optional).
        :param n: Number of texts to retrieve (default: 9).
        :param plot_image: Plot the query
        :param temperature: change real distribution, default = 2.5
        :return: List of retrieved text samples and its probabilities.
        """
        if text_embeddings is None:
            text_embeddings = self.encode_text(texts)

        image_embeddings = self.encode_image([query])
        image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
        dot_similarity = (image_embeddings_n @ text_embeddings_n.T) * torch.exp(
            torch.tensor(temperature).to(self.device))

        if n > len(texts):
            n = len(texts)

        values, indices = torch.topk(dot_similarity.cpu().squeeze(0), n)
        matches = [texts[idx] for idx in indices]
        if plot_image:
            # Read and plot the image
            image = self.load_image(query)
            # Plot the image
            plt.imshow(image)
            plt.title('Random Image')
            plt.axis('off')
            plt.show()
        return matches, values

    def forward(self, inputs):
        """
        Forward pass of the model.

        :param inputs: Input dictionary containing 'image', 'input_ids', and 'attention_mask'.
        :return: Loss value.
        """
        images = inputs['image']
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # For constant temperature replace self.temperature by self.temperature.data and give the target temperature
        # for example temperature = 1.0 or temperature = 2.5
        t = torch.clamp(self.temperature, min=torch.tensor(0.01).to(self.device),
                        max=torch.tensor(20).to(self.device))

        # Embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder([input_ids, attention_mask])
        image_context_feature = self.context_encoder(torch.tensor([self.image_context_id],
                                                                  dtype=torch.float32).unsqueeze(1).to(self.device))
        text_context_feature = self.context_encoder(torch.tensor([self.text_context_id],
                                                                 dtype=torch.float32).unsqueeze(1).to(self.device))
        # repeat: images.size for the first dim and 1 for the second dim (like tile in tf)
        image_context_features = image_context_feature.repeat(images.size(0), 1)
        text_context_features = text_context_feature.repeat(images.size(0), 1)
        # Fusion
        image_features = self.fusion_encoder([image_features, image_context_features]).average_outputs
        text_features = self.fusion_encoder([text_features, text_context_features]).average_outputs
        # L2 normalization
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        # Compute loss
        logits = (image_features @ text_features.T) * torch.exp(t)
        # labels
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_t) / 2.0

        return loss

    def __call__(self, inputs):
        return self.forward(inputs)


def get_image_embeddings(pairs, model_path):
    # Process pairs to delete duplicated images
    unique_images = set()
    unique_pairs = [(item[0], item[1]) for item in pairs if item[0] not in unique_images
                    and not unique_images.add(item[0])]
    # sort images
    unique_pairs = sorted(unique_pairs, key=lambda x: x[0])
    # Build model
    model = Model(device=CFG.device)
    # Load parameters
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    valid_loader = build_loaders(unique_pairs, model.tokenizer, mode="valid")
    # Use eval mode to freeze all layers
    model.eval()
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            contexts = torch.tensor([1], dtype=torch.float32).unsqueeze(1).to(CFG.device)
            context_embedding = model.context_encoder(contexts)
            context_embeddings = context_embedding.repeat(batch["image"].size(0), 1)
            image_embeddings = model.image_encoder(batch["image"].to(CFG.device))
            fusion_embeddings = model.fusion_encoder([image_embeddings, context_embeddings]).average_outputs
            valid_image_embeddings.append(fusion_embeddings)
    return torch.cat(valid_image_embeddings)


def get_caption_embeddings(pairs, model_path):
    # sort according images
    unique_pairs = sorted(pairs, key=lambda x: x[0])
    # Build model
    model = Model(device=CFG.device)
    # Load parameters
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    valid_loader = build_loaders(unique_pairs, model.tokenizer, mode="valid")
    # Use eval mode to freeze all layers
    model.eval()
    valid_caption_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            contexts = torch.tensor([0], dtype=torch.float32).unsqueeze(1).to(CFG.device)
            context_embedding = model.context_encoder(contexts)
            context_embeddings = context_embedding.repeat(batch["image"].size(0), 1)
            caption_embeddings = model.text_encoder([batch["input_ids"].to(CFG.device),
                                                   batch["attention_mask"].to(CFG.device)])
            fusion_embeddings = model.fusion_encoder([caption_embeddings, context_embeddings]).average_outputs
            valid_caption_embeddings.append(fusion_embeddings)
    return torch.cat(valid_caption_embeddings)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    # List of dictionary, each dictionary is a batch
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:

        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter
