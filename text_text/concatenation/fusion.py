import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
from configs import CFG


class ContextEncoder(nn.Module):
    def __init__(self, input_dim=CFG.context_input_dim,dropout_rate=CFG.dropout_rate, device=CFG.device,
                 output_dim=CFG.context_output_dim, *args, **kwargs):
        super(ContextEncoder, self).__init__(*args, **kwargs)
        # Attributes
        self.context_input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.device = device
        self.context_output_dim = output_dim
        # Models
        roberta_variance = torch.rand(1) * 0.5 + 0.1
        bert_variance = torch.rand(1) * 0.5 + 0.1
        self.text_context = nn.Parameter(torch.normal(mean=0, std=roberta_variance.item(),
                                                      size=(1, self.context_input_dim)).to(self.device))
        self.image_context = nn.Parameter(torch.normal(mean=0, std=bert_variance.item(),
                                                       size=(1, self.context_input_dim)).to(self.device))
        self.model = nn.Sequential(
            nn.Linear(self.context_input_dim, self.context_output_dim),
            nn.LayerNorm(self.context_output_dim)
        )

    def forward(self, inputs):
        # 0: roberta and 1: bert
        x = inputs
        context = torch.where(x == 0, self.text_context, self.image_context)
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
    def __init__(self, input_dim=CFG.projection_dim, num_head=CFG.num_head, num_layers=CFG.num_layers,
                 context_output_dm=CFG.context_output_dim, *args, **kwargs):
        """
         Initialize Fusion module.

            :param input_dim: Dimensionality of the input embeddings. Defaults to CFG.projection_dim.
            :param num_head: Number of attention heads. Defaults to CFG.num_head.
            :param num_layers: Number of transformer layers. Defaults to CFG.num_layers.
            :param context_output_dm: Dimension of the output dimension
        """
        super(Fusion, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_head = num_head
        self.num_layers = num_layers
        self.projection_dim = input_dim
        self.context_output_dim = context_output_dm
        # Models and layer
        self.projection = nn.Sequential(
            # Concatenation of projection_dim give projection_dim*2
            nn.Linear(self.projection_dim+self.context_output_dim, self.projection_dim+self.context_output_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim+self.context_output_dim, self.projection_dim),
        )
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
        # x: bert or roberta embeddings
        x, contexts = inputs

        ## Fusion block
        contexts = contexts.unsqueeze(1).expand(-1, x.size()[1], -1)
        # Concatenate contexts with image/caption embeddings
        concat = torch.cat((x, contexts), dim=-1)
        concat = self.projection(concat)

        # Residual connection
        residual_connection = x + concat

        # Normalization
        norm = self.layer_normalization(residual_connection)

        # Projection
        encoder_output = self.transformer_encoder(norm)

        # Normalization
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
