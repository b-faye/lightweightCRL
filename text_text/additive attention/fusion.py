import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
from configs import CFG
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, *args, **kwargs):
        super(BahdanauAttention, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.value_projection = nn.Linear(self.input_dim, self.input_dim)
        self.context_projection = nn.Linear(self.input_dim, self.input_dim)
        self.linear_projection = nn.Linear(self.input_dim, 1)

    def forward(self, inputs):
        # (batch_size, sequence, projection_dim), (batch_size, projection_dim)
        values, contexts = inputs
        # output: (batch_size, sequence, 1)
        scores = self.linear_projection(torch.tanh(self.context_projection(contexts) + self.value_projection(values)))
        attention_weights = F.softmax(scores, dim=1)
        # output: (batch_size, sequence, projection_dim)
        bahdanau_contexts = attention_weights * values
        # output: (batch_size, projection_dim)
        return bahdanau_contexts.sum(dim=1)

    def __call__(self, inputs):
        return self.forward(inputs)


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=CFG.context_input_dim, projection_dim=CFG.projection_dim,
                 dropout_rate=CFG.dropout_rate, device=CFG.device, *args, **kwargs):
        super(ContextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.context_input_dim = input_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.device = device
        # Models
        roberta_variance = torch.rand(1) * 0.5 + 0.1
        bert_variance = torch.rand(1) * 0.5 + 0.1
        self.roberta_context = nn.Parameter(torch.normal(mean=0, std=roberta_variance.item(),
                                                      size=(1, self.context_input_dim)).to(self.device))
        self.bert_context = nn.Parameter(torch.normal(mean=0, std=bert_variance.item(),
                                                       size=(1, self.context_input_dim)).to(self.device))
        self.model = nn.Sequential(
            nn.Linear(self.context_input_dim, 64),  # Adjust based on your requirements
            nn.ReLU(),
            nn.Linear(64, 128),  # Adjust based on your requirements
            nn.ReLU(),
            nn.Linear(128, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )

    def forward(self, inputs):
        # 0: roberta and 1: bert
        x = inputs
        context = torch.where(x == 0, self.roberta_context, self.bert_context)
        # output : torch.Size([batch_size, 256])
        return self.model(context)

    def __call__(self, inputs):
        return self.forward(inputs)


class FusionOutput:
    def __init__(self, outputs):
        """
        Wrapper class for fusion model outputs.

        :param outputs: Dictionary containing model outputs.
        """
        self.outputs = outputs

    def __getattr__(self, name):
        """
        Retrieve attribute from outputs dictionary.

        :param name: Name of the attribute to retrieve.
        :return: Value of the attribute.
        """
        if name in self.outputs:
            return self.outputs[name]
        else:
            raise AttributeError(f"'FusionOutput' object has no attribute '{name}'")


class Fusion(nn.Module):
    def __init__(self, input_dim=CFG.projection_dim, num_head=CFG.num_head, num_layers=CFG.num_layers, *args, **kwargs):
        """
         Initialize Fusion module.

            :param input_dim: Dimensionality of the input embeddings. Defaults to CFG.projection_dim.
            :param num_head: Number of attention heads. Defaults to CFG.num_head.
            :param num_layers: Number of transformer layers. Defaults to CFG.num_layers.
        """
        super(Fusion, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.projection_dim = input_dim
        # Models and layers
        self.bahdanau_attention = BahdanauAttention(self.input_dim)
        self.transformer_encoder_block = nn.TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=self.num_head,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_block,
            num_layers=self.num_layers
        )

        self.layer_normalization = nn.LayerNorm(self.projection_dim)

    def forward(self, inputs):
        x, contexts = inputs

        # Fusion block
        attention = self.bahdanau_attention([x, contexts.unsqueeze(1)])
        attention = attention.unsqueeze(1)

        # Residual connection
        residual_connection = x + attention

        # Normalization
        norm = self.layer_normalization(residual_connection)

        # Projection
        encoder_output = self.transformer_encoder(norm)

        # Residual connection
        residual_connection = residual_connection + encoder_output

        # Possible outputs
        sequence_outputs = residual_connection
        average_outputs = torch.mean(residual_connection, dim=1)
        max_outputs = torch.max(residual_connection, dim=1)
        min_outputs = torch.min(residual_connection, dim=1)
        return FusionOutput({'sequence_outputs': sequence_outputs, 'average_outputs': average_outputs,
                             'max_outputs': max_outputs, 'min_outputs': min_outputs})

    def __call__(self, inputs):
        return self.forward(inputs)
