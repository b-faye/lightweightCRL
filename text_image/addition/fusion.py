import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
from configs import CFG


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=CFG.context_input_dim, projection_dim=CFG.projection_dim,
                 dropout_rate=CFG.dropout_rate, device='cpu', *args, **kwargs):
        """
        Context encoder module for encoding context information.

        :param input_dim: Dimensionality of the input context (default: CFG.context_input_dim).
        :param projection_dim: Dimensionality of projected features (default: CFG.projection_dim).
        :param dropout_rate: Dropout rate (default: CFG.dropout_rate).
        :param device: Device to run the module on (default: 'cpu').
        """
        super(ContextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.context_input_dim = input_dim
        self.projection_dim = projection_dim
        self.dropout_rate = dropout_rate
        self.device = device
        # Models
        text_variance = torch.rand(1) * 0.5 + 0.1
        image_variance = torch.rand(1) * 0.5 + 0.1
        self.text_context = nn.Parameter(torch.normal(mean=0, std=text_variance.item(),
                                                      size=(1, self.context_input_dim)).to(self.device))
        self.image_context = nn.Parameter(torch.normal(mean=0, std=image_variance.item(),
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
        """
        Forward pass of the context encoder.

        :param inputs: Input context indicator (0 for text, 1 for image).
        :return: Projected features.
        """
        # 0: text and 1: image
        x = inputs
        context = torch.where(x == 0, self.text_context, self.image_context)
        # output: torch.Size([batch_size, 256])
        return self.model(context)

    def __call__(self, inputs):
        """
        Callable method for the context encoder.

        :param inputs: Input context indicator (0 for text, 1 for image).
        :return: Projected features.
        """
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
        # x: image or caption embeddings
        x, contexts = inputs

        ## Fusion block
        addition = x + contexts.unsqueeze(1)

        # Normalization
        norm = self.layer_normalization(addition)

        # Projection
        encoder_output = self.transformer_encoder(norm)

        # Residual connection
        residual_connection = addition + encoder_output

        # Possible outputs
        sequence_outputs = residual_connection
        average_outputs = torch.mean(residual_connection, dim=1)
        max_outputs = torch.max(residual_connection, dim=1)
        min_outputs = torch.min(residual_connection, dim=1)
        return FusionOutput({'sequence_outputs': sequence_outputs, 'average_outputs': average_outputs,
                             'max_outputs': max_outputs, 'min_outputs': min_outputs})

    def __call__(self, inputs):
        return self.forward(inputs)
