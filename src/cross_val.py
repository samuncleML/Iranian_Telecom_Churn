import os
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchmetrics import F1Score, Accuracy, Recall, Precision
from dataset import cross_val_train, cross_val_validation, BASE_PATH
from module import ChurnModule

f1_score = F1Score(task='binary')
accuracy_score = Accuracy(task='binary')
recall_score = Recall(task='binary')
precision_score = Precision(task='binary')

f1_score_val = F1Score(task='binary')
accuracy_score_val = Accuracy(task='binary')
recall_score_val = Recall(task='binary')
precision_score_val = Precision(task='binary')

model = ChurnModule()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 150
for train_loader, val_loader in zip(cross_val_train, cross_val_validation):
    for epoch in range(1, EPOCHS+1):
        running_loss = 0.0

        for features, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            outputs = model(features)

            loss = criterion(outputs.squeeze(1), labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        f1_score_val.reset()
        accuracy_score_val.reset()
        recall_score_val.reset()
        precision_score_val.reset()

        model.eval()
        with torch.no_grad():
            for feature, label in val_loader:
                optimizer.zero_grad()
                output = model(feature)

                loss = criterion(output.squeeze(1), label)
                f1_score_val.update(output.squeeze(1), label)
                accuracy_score_val.update(output.squeeze(1), label)
                recall_score_val.update(output.squeeze(1), label)

                precision_score_val.update(output.squeeze(1), label)
                running_loss += loss.item()

            accuracy = accuracy_score_val.compute().item()
            recall = recall_score_val.compute().item()
            precision = precision_score_val.compute().item()
            f1 = f1_score_val.compute().item()
            
    print(f'Validation -----> Loss {(running_loss/len(val_loader)):.2f} | Accuracy - {accuracy*100:.2f}% | F1 - {f1*100:.2f}% | Recall - {recall*100:.2f}% | Precision - {precision*100:.2f}%')
torch.save(model.state_dict(), os.path.join(BASE_PATH, 'models', 'Iranian_Telecom_Churn_CV_model.pth'))