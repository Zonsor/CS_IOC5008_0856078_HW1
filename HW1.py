# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:40:37 2019

@author: Zonsor
"""
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.utils.data import DataLoader

from HW1Dataset import HW1TestDataset
from pathlib import Path
import csv


def train(train_path, data_transforms, load_pkl):
    hw1_dataset_train = datasets.ImageFolder(train_path,
                                             data_transforms)
    train_loader = DataLoader(
                dataset=hw1_dataset_train,
                batch_size=20,
                shuffle=True,
                )

    if load_pkl:
        model = torch.load('HW1_ResNext101')
    else:
        model = models.resnext101_32x8d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 13)

    model = model.cuda()
    model.train()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.001,
                                momentum=0.9)

    for epoch in range(50):
        print('Epoch: {:d}'.format(epoch+1))
        print('-' * len('Epoch: {:d}'.format(epoch+1)))
        train_loss = 0.0
        train_corrects = 0

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()
            prediction = model(x_batch)
            _, preds = torch.max(prediction, 1)
            loss = loss_func(prediction, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            train_corrects += torch.sum(preds == y_batch.data)

        train_loss = train_loss / len(hw1_dataset_train)
        train_acc = train_corrects.double() / len(hw1_dataset_train)
        print('Training loss: {:.4f}\taccuracy: {:.4f}\n'
              .format(train_loss, train_acc))
        if epoch == 0:
            best_loss = train_loss
        elif train_loss < best_loss:
            best_loss = train_loss
            torch.save(model, 'HW1_ResNext101.pkl')


def test(test_path, data_transforms):
    model = torch.load('HW1_ResNext101.pkl')
    model = model.cuda()
    model.eval()

    dataset_test = HW1TestDataset(test_path, data_transforms)
    test_loader = DataLoader(
                dataset=dataset_test,
                batch_size=1,
                )
    classes = [_dir.name for _dir in Path(train_path).glob('*')]
    table = [['id', 'label']]
    with torch.no_grad():
        for test_input, filename in test_loader:
            test_input = test_input.cuda()
            test_output = model(test_input)
            _, prediction = torch.max(test_output, 1)
            table.append([filename[0][:-4], classes[prediction.item()]])

    with open('HW1_test_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(table)


if __name__ == '__main__':
    train_path = 'dataset/train'
    test_path = 'dataset/test'
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(8),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }
    load_pkl = False

    train(train_path, data_transforms['train'], load_pkl)
    test(test_path, data_transforms['test'])
