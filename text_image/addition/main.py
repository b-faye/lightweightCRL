import os
import sys
package_dir = os.getcwd()
root_dir = os.path.dirname(package_dir)
sys.path.append(root_dir)
import torch
from configs import CFG
from coco_dataset import download_dataset, make_pairs, build_loaders, make_positive_pairs
import random
from model import Model
import itertools
from model import train_epoch, valid_epoch


def main(n_pairs=500000, basic_train=False, download_coco=False):
    if download_coco:
        download_dataset()

    # Create pairs image-caption
    if not os.path.exists(os.path.join(CFG.data_directory, "similarity_data_train.pt")):
        """
        Number of training images: 2.164.546
        Number of validation images: 381.979
        """

        training_pairs = make_positive_pairs(CFG.train_annotation_file)
        torch.save(training_pairs, os.path.join(CFG.data_directory, "similarity_data_train.pt"))
        validation_pairs = make_positive_pairs(CFG.val_annotation_file)
        torch.save(validation_pairs, os.path.join(CFG.data_directory, "similarity_data_val.pt"))
    else:
        training_pairs = torch.load(os.path.join(CFG.data_directory, "similarity_data_train.pt"))
        validation_pairs = torch.load(os.path.join(CFG.data_directory, "similarity_data_val.pt"))
    # number of pairs for training
    random.shuffle(training_pairs)
    random.shuffle(validation_pairs)
    training_pairs = training_pairs[:n_pairs]
    validation_pairs = validation_pairs[-round(len(training_pairs)*0.20):]
    print("Number of training images: {}".format(len(training_pairs)))
    print("Number of validation images: {}".format(len(validation_pairs)))
    # Build loader : return dictionary
    train_loader = build_loaders(training_pairs)
    val_loader = build_loaders(validation_pairs)
    model = Model(device=CFG.device)
    if basic_train:
        # Train all parameters with the same lr and weight decay
        # This method is better when using dynamique temperature
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=CFG.weight_decay, lr=CFG.lr)
    else:
        parameters = [
            {"params": model.roberta_encoder.pretrained_roberta.parameters(), "lr": CFG.roberta_encoder_lr},
            {"params": model.bert_encoder.pretrained_bert.parameters(), "lr": CFG.bert_encoder_lr},
            {
                "params": itertools.chain(
                    model.context_encoder.parameters(), model.fusion_encoder.parameters(),
                    model.roberta_encoder.projection_head.parameters(), model.bert_encoder.projection_head.parameters()
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
    # Train the model
    for epoch in range(CFG.epochs):
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
            torch.save(model.state_dict(), f"best.pt")
            print("Saved best model!")
        else:
            if epoch >= CFG.patience - 1:
                num_bad_epochs += 1
            if num_bad_epochs >= CFG.patience:
                print(f"Early stopping at epoch {epoch + 1}. Restoring best weights...")
                break
        lr_scheduler.step(val_loss.avg)
    torch.save(model.state_dict(), "last.pt")
    # Free GPU
    model = None
    optimizer = None
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
