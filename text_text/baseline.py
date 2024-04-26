import torch.nn as nn
import timm
from transformers import AutoTokenizer, BertModel, RobertaModel, RobertaConfig
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


class RobertaEncoder(nn.Module):
    def __init__(self, model_name=CFG.roberta_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        super(RobertaEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        config = RobertaConfig.from_pretrained(self.model_name, use_pooler=False)
        self.pretrained_roberta = RobertaModel.from_pretrained(self.model_name, config=config)
        self.projection_head = ProjectionHead(self.pretrained_roberta.config.hidden_size,
                                              self.projection_dim, self.dropout_rate)
        # Freeze bert
        for parameter in self.pretrained_roberta.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        # last_hidden_state: torch.Size([batch_size, sequence, 768])
        # pooler_output: torch.Size([batch_size, 768])
        x = self.pretrained_roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # output: torch.Size([batch_size, sequence, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)


class BertEncoder(nn.Module):
    def __init__(self, model_name=CFG.bert_name, projection_dim=CFG.projection_dim,
                 trainable=False, dropout_rate=CFG.dropout_rate, *args, **kwargs):
        super(BertEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.trainable = trainable
        # Models
        self.pretrained_bert = BertModel.from_pretrained(self.model_name)
        self.projection_head = ProjectionHead(self.pretrained_bert.config.hidden_size,
                                              self.projection_dim, self.dropout_rate)
        # Freeze bert
        for parameter in self.pretrained_bert.parameters():
            parameter.requires_grad = self.trainable

    def forward(self, inputs):
        input_ids, attention_mask = inputs
        # last_hidden_state: torch.Size([batch_size, sequence, 768])
        # pooler_output: torch.Size([batch_size, 768])
        x = self.pretrained_bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # output: torch.Size([batch_size, sequence, 256])
        x = self.projection_head(x)
        return x

    def __call__(self, inputs):
        return self.forward(inputs)
