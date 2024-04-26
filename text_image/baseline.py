import torch.nn as nn
import timm
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertConfig, DistilBertTokenizer
from configs import CFG


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim=CFG.projection_dim, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Projection Head module for contrastive learning.

        :param input_dim: Dimensionality of input features.
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(ProjectionHead, self).__init__(*args, **kwargs)
        # Attributes
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        # Layers
        self.linear_layer1 = nn.Linear(self.input_dim, self.projection_dim)
        self.gelu = nn.GELU()
        self.linear_layer2 = nn.Linear(self.projection_dim, self.projection_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.normalization_layer = nn.LayerNorm(self.projection_dim)

    def forward(self, inputs):
        """
        Forward pass of the projection head.

        :param inputs: Input features.
        :return: Projected features.
        """
        x = inputs
        x = self.linear_layer1(x)
        x = self.gelu(x)
        x = self.linear_layer2(x)
        x = self.dropout(x)
        x = self.normalization_layer(x)
        return x

    def __call__(self, inputs):
        """
        Callable method for the projection head.

        :param inputs: Input features.
        :return: Projected features.
        """
        return self.forward(inputs)


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.vit_name, projection_dim=CFG.projection_dim, trainable=False,
                 dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Image encoder module using Vision Transformer (ViT) backbone.

        :param model_name: Name of the Vision Transformer model (default: CFG.vit_name).
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param trainable: Whether to make the backbone trainable (default: False).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(ImageEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        # Models
        self.pretrained_vit = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self.projection_head = ProjectionHead(self.pretrained_vit.embed_dim, self.projection_dim, self.dropout_rate)
        # Freeze pretrained ViT layers
        for parameter in self.pretrained_vit.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, images):
        """
        Forward pass of the image encoder.

        :param images: Input images.
        :return: Projected features.
        """
        x = images
        # forward_features: to return sequences (encoder) -> torch.Size([batch_size, 197, 768]) forward_head: to
        # return flattened sequences (vectors) -> torch.Size([batch_size, 768]) if num_classes=0 (no classification)
        # in timm.create_model and torch.Size([batch_size, 1000]) otherwise (classification)
        x = self.pretrained_vit.forward_features(x)
        # output: torch.Size([batch_size, 197, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, images):
        """
        Callable method for the image encoder.

        :param images: Input images.
        :return: Projected features.
        """
        return self.forward(images)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.bert_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        """
        Text encoder module using BERT backbone.

        :param model_name: Name of the BERT model (default: CFG.bert_name).
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param trainable: Whether to make the backbone trainable (default: False).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        """
        super(TextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_bert = BertModel.from_pretrained(self.model_name)
        self.projection_head = ProjectionHead(self.pretrained_bert.config.hidden_size,
                                              self.projection_dim, self.dropout_rate)
        # Freeze BERT
        for parameter in self.pretrained_bert.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, captions):
        """
        Forward pass of the text encoder.

        :param captions: Input captions (input_ids, attention_mask).
        :return: Projected features.
        """
        input_ids, attention_mask = captions
        # last_hidden_state: torch.Size([batch_size, sequence, 768])
        # pooler_output: torch.Size([batch_size, 768])
        x = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # output: torch.Size([batch_size, sequence, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, captions):
        """
        Callable method for the text encoder.

        :param captions: Input captions (input_ids, attention_mask).
        :return: Projected features.
        """
        return self.forward(captions)
