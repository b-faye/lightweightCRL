import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
from configs import CFG
from transformers import BertTokenizer
from coco_dataset import download_dataset, make_pairs, build_loaders
import random
from model import Model
import itertools
from model import train_epoch, valid_epoch, get_image_embeddings, get_caption_embeddings


def main(basic_train=False):
    # Load the pretrained Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained(CFG.bert_name)
    # Load COCO dataset if not exist
    if len(os.listdir(CFG.data_directory)) == 0:
        download_dataset()
    # Create pairs image-caption 413.915
    training_pairs = make_pairs(CFG.train_annotation_file, CFG.image_dir, 5)
    random.shuffle(training_pairs)
    # validation 202.520
    validation_pairs = make_pairs(CFG.val_annotation_file, CFG.image_dir_val, 5)
    random.shuffle(validation_pairs)
    validation_pairs = validation_pairs[-round(len(validation_pairs)*0.20):]
    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))
    # Build loader : return dictionary
    train_loader = build_loaders(training_pairs, tokenizer, mode="train")
    val_loader = build_loaders(validation_pairs, tokenizer, mode="valid")
    # Create the training model
    model = Model(device=CFG.device)
    if basic_train:
        # Train all parameters with the same lr and weight decay
        # This method is better when using dynamique temperature
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.image_encoder.pretrained_vit.parameters(), "lr": CFG.image_encoder_lr},
            {"params": model.text_encoder.pretrained_bert.parameters(), "lr": CFG.text_encoder_lr},
            {
                "params": itertools.chain(
                    model.context_encoder.parameters(), model.fusion_encoder.parameters(),
                    model.image_encoder.projection_head.parameters(), model.text_encoder.projection_head.parameters()
                ),
                "lr": CFG.lr, "weight_decay": CFG.weight_decay
            },
            {"params": [model.temperature], "lr": CFG.lr, "weight_decay": CFG.weight_decay}
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=0.)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    num_bad_epochs = 0
    best_loss = float('inf')
    best_epoch = 0

    # Train the model
    for epoch in range(CFG.epochs):
        print(model.temperature.data)
        print("Epoch: %d" % (epoch+1))
        # Set the model in train mode
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, "epoch")
        print(f"Epoch: {epoch+1}, train loss: {train_loss}")
        # Set the model in evaluation mode
        model.eval()
        with torch.no_grad():
            val_loss = valid_epoch(model, val_loader)
            print(f"Epoch: {epoch + 1}, val loss: {val_loss}")
        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            num_bad_epochs = 0
            torch.save(model.state_dict(), "best.pt")
            best_epoch = epoch+1
            print("Saved best model!")
        else:
            if epoch >= CFG.patience - 1:
                num_bad_epochs += 1
            if num_bad_epochs >= CFG.patience:
                print(f"Early stopping at epoch {epoch + 1}. Restoring best weights...")
                break
        lr_scheduler.step(val_loss.avg)
    torch.save(model.state_dict(), "last.pt")
    # Save train embeddings with best.pt
    image_embeddings = get_image_embeddings(training_pairs, "best.pt")
    torch.save(image_embeddings, "image_embeddings_best.pt")
    caption_embeddings = get_caption_embeddings(training_pairs, "best.pt")
    torch.save(caption_embeddings, "caption_embeddings_best.pt")
    # Free GPU
    model = None
    optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
