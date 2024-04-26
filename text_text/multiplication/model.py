import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
import torch.nn as nn
from baseline import ImageEncoder, BertEncoder, RobertaEncoder
from fusion import Fusion, ContextEncoder
from transformers import AutoTokenizer
from configs import CFG
import torch.nn.functional as F
from tqdm import tqdm
from loss import AvgMeter


class Model(nn.Module):

    def __init__(self, context_input_dim=CFG.context_input_dim, device='cpu',
                 roberta_tokenizer=AutoTokenizer.from_pretrained(CFG.roberta_name),
                 bert_tokenizer=AutoTokenizer.from_pretrained(CFG.bert_name),
                 token_max_length=CFG.max_length, roberta_encoder=RobertaEncoder(),
                 bert_encoder=BertEncoder(), context_encoder=ContextEncoder(),
                 fusion_encoder=Fusion(), *args, **kwargs):

        super(Model, self).__init__(*args, **kwargs)
        self.context_input_dim = context_input_dim
        self.device = device
        self.to(self.device)
        self.roberta_tokenizer = roberta_tokenizer
        self.roberta_encoder = roberta_encoder.to(self.device)
        self.bert_encoder = bert_encoder.to(self.device)
        self.bert_tokenizer = bert_tokenizer
        self.context_encoder = context_encoder.to(self.device)
        self.context_encoder.device = self.device
        self.fusion_encoder = fusion_encoder.to(self.device)
        self.roberta_context_id = 0
        self.bert_context_id = 1
        self.temperature = nn.Parameter(torch.tensor(0.07).to(self.device))
        self.max_length = token_max_length

    def matching(self, text_1, text_2, model_1="bert_fusion", model_2="roberta_fusion"):
        """
        :param text_1: list of texts
        :param text_2: list of texts
        :param model_1: must be bert_baseline, roberta_baseline, bert_fusion or roberta_fusion
        :param model_2: must be bert_baseline, roberta_baseline, bert_fusion or roberta_fusion
        :return: similarity scores
        """
        text_1_features = self.encode_texts(texts=text_1, method=model_1)
        text_2_features = self.encode_texts(texts=text_2, method=model_2)

        text_1_features = F.normalize(text_1_features, p=2, dim=-1)
        text_2_features = F.normalize(text_2_features, p=2, dim=-1)

        scores = text_1_features @ text_2_features.T

        return scores

    def encode_texts(self, texts, method="bert_fusion", strategy="average"):
        """
        :param method: must be bert_baseline, roberta_baseline, bert_fusion or roberta_fusion
        :param texts: list of texts
        :param strategy: must be average, min, max or sequence
        :return: encoded text features
        """
        if method == "bert_baseline":
            text_tokenized = self.bert_tokenizer(texts, padding=True, truncation=True,
                                                 max_length=self.max_length)
            text_features = self.bert_encoder.pretrained_bert(
                input_ids=torch.tensor(text_tokenized["input_ids"]).to(self.device),
                attention_mask=torch.tensor(text_tokenized["attention_mask"]).to(self.device)
            )
            if strategy == "sequence":
                return text_features.last_hidden_state
            else:
                return text_features.pooler_output

        elif method == "roberta_baseline":
            text_tokenized = self.roberta_tokenizer(texts, padding=True, truncation=True,
                                                    max_length=self.max_length)
            text_features = self.roberta_encoder.pretrained_roberta(
                input_ids=torch.tensor(text_tokenized["input_ids"]).to(self.device),
                attention_mask=torch.tensor(text_tokenized["attention_mask"]).to(self.device)
            )
            if strategy == "sequence":
                return text_features.last_hidden_state
            else:
                return text_features.pooler_output

        elif method == "roberta_fusion":
            text_tokenized = self.roberta_tokenizer(texts, padding=True, truncation=True,
                                                    max_length=self.max_length)
            text_features = self.roberta_encoder([
                torch.tensor(text_tokenized["input_ids"]).to(self.device),
                torch.tensor(text_tokenized["attention_mask"]).to(self.device)
            ])
            roberta_context_feature = self.context_encoder(
                torch.tensor([self.roberta_context_id], dtype=torch.float32).unsqueeze(1).to(self.device)
            )
            roberta_context_features = roberta_context_feature.repeat(text_features.size(0), 1)
            text_features = self.fusion_encoder([text_features, roberta_context_features])

            if strategy == "sequence":
                return text_features.sequence_outputs
            elif strategy == "min":
                return text_features.min_outputs
            elif strategy == "max":
                return text_features.max_outputs
            else:
                return text_features.average_outputs

        else:
            text_tokenized = self.bert_tokenizer(texts, padding=True, truncation=True,
                                                 max_length=self.max_length)
            text_features = self.bert_encoder([
                torch.tensor(text_tokenized["input_ids"]).to(self.device),
                torch.tensor(text_tokenized["attention_mask"]).to(self.device)
            ])
            bert_context_feature = self.context_encoder(
                torch.tensor([self.bert_context_id], dtype=torch.float32).unsqueeze(1).to(self.device)
            )
            bert_context_features = bert_context_feature.repeat(text_features.size(0), 1)
            text_features = self.fusion_encoder([text_features, bert_context_features])

            if strategy == "sequence":
                return text_features.sequence_outputs
            elif strategy == "min":
                return text_features.min_outputs
            elif strategy == "max":
                return text_features.max_outputs
            else:
                return text_features.average_outputs

    def forward(self, inputs):
        # Data
        roberta_texts = list(inputs[0])
        bert_texts = list(inputs[1])
        tokenized_roberta_texts = self.roberta_tokenizer(roberta_texts, padding=True, truncation=True,
                                                         max_length=self.max_length)
        tokenized_bert_texts = self.bert_tokenizer(bert_texts, padding=True, truncation=True,
                                                   max_length=self.max_length)

        # For constant temperature replace self.temperature by self.temperature.data and give the target temperature
        # for example temperature = 1.0 or temperature = 2.5
        t = torch.clamp(self.temperature, min=torch.tensor(0.01).to(self.device),
                        max=torch.tensor(20).to(self.device))

        # Embeddings
        roberta_features = self.roberta_encoder([torch.tensor(tokenized_roberta_texts['input_ids']).to(self.device),
                                               torch.tensor(tokenized_roberta_texts['attention_mask']).to(self.device)])
        bert_features = self.bert_encoder([torch.tensor(tokenized_bert_texts['input_ids']).to(self.device),
                                             torch.tensor(tokenized_bert_texts['attention_mask']).to(self.device)])

        # Contexts
        roberta_context_feature = self.context_encoder(torch.tensor([self.roberta_context_id],
                                                                     dtype=torch.float32).unsqueeze(1).to(self.device))
        roberta_context_features = roberta_context_feature.repeat(roberta_features.size(0), 1)
        bert_context_feature = self.context_encoder(torch.tensor([self.bert_context_id],
                                                                     dtype=torch.float32).unsqueeze(1).to(self.device))
        bert_context_features = bert_context_feature.repeat(bert_features.size(0), 1)

        # Fusion
        roberta_features = self.fusion_encoder([roberta_features, roberta_context_features])
        bert_features = self.fusion_encoder([bert_features, bert_context_features])

        # L2 normalization
        roberta_features = F.normalize(roberta_features.average_outputs, p=2, dim=-1)
        bert_features = F.normalize(bert_features.average_outputs, p=2, dim=-1)

        # Compute loss
        logits = (roberta_features @ bert_features.T) * torch.exp(t)

        # labels
        labels = torch.arange(logits.size(0)).to(self.device)
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_t = F.cross_entropy(logits.t(), labels, reduction='mean')
        loss = (loss_i + loss_t) / 2.0

        return loss


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
