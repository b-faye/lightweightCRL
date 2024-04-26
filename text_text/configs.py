import torch
import os


class CFG:
    max_length = 128
    batch_size = 64
    num_workers = 4
    projection_dim = 256
    dropout_rate = 0.1
    context_input_dim = 3
    num_head = 4
    num_layers = 1
    roberta_encoder_lr = 1e-4
    bert_encoder_lr = 1e-5
    context_encoder_lr = 1e-3
    fusion_lr = 1e-3
    lr = 1e-3
    context_output_dim = 10
    weight_decay = 1e-3
    patience = 5
    factor = 0.8
    epochs = 100
    data_directory = "../datasets"
    train_annotation_file = os.path.join(data_directory, "annotations", "captions_train2014.json")
    val_annotation_file = os.path.join(data_directory, "annotations", "captions_val2014.json")
    image_dir = os.path.join(data_directory, "train2014")
    image_dir_val = os.path.join(data_directory, "val2014")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    roberta_name = "roberta-base"
    bert_name = "bert-base-uncased"
