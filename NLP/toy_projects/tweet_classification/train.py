import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from dataloader import CustomDataset, load_data
from model import Model
import json
from dataloader import load_data


##import config file


def train(model, train_loader):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for idx, batch in enumerate(train_loader):
        input, label = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        prediction = model(input)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(prediction.data, 1)

        train_loss += loss.item()
        train_acc += (predicted == label).sum().item()

        total_train_acc = (train_acc / len(train_loader.dataset)) * 100
        avg_loss = train_loss / len(train_loader)

    return avg_loss, total_train_acc


def validate(model, valid_loader):
    val_loss = 0.0
    val_acc = 0.0

    for idx, batch in enumerate(valid_loader):
        input, label = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()

        prediction = model(input)
        loss = criterion(prediction, label)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(prediction.data, 1)

        val_loss += loss.item()
        val_acc += (predicted == label).sum().item()

        total_val_acc = (val_acc / len(valid_loader.dataset)) * 100
        avg_loss = val_loss / len(valid_loader)

    return avg_loss, total_val_acc


if __name__ == "__main__":
    with open("./config/train.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda")

    train_loader, valid_loader, vocab_size = load_data(config, "train")
    model = Model(**config['Model'], vocab_size=vocab_size, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['Train']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['Train']['epoch']):
        train_loss, train_acc = train(model, train_loader)
        val_loss, val_acc = validate(model, valid_loader)

        best_val_acc = 0.0

        if best_val_acc < val_acc:
            best_val_ep = epoch + 1
            torch.save(model.state_dict(), "./ckpt/tweet_classification_ckpt_" + str(best_val_ep))
            best_val_acc = val_acc

        print("############ EPOCH {} ############".format(epoch + 1))
        print("train loss: {:.4f} , val loss : {:.4f}".format(train_loss, val_loss))
        print("train acc: {:.4f} , val acc : {:.4f}".format(train_acc, val_acc))
        print("\n")
