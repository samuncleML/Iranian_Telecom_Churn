import torch
import torch.optim as optim
import torch.nn as nn
from torchmetrics import F1Score, Accuracy, Recall, Precision
from dataset import train_loader, val_loader
from module import ChurnModule

f1_score = F1Score(task='binary')
accuracy_score = Accuracy(task='binary')
recall_score = Recall(task='binary')
precision_score = Precision(task='binary')

f1_score_ = F1Score(task='binary')
accuracy_score_ = Accuracy(task='binary')
recall_score_ = Recall(task='binary')
precision_score_ = Precision(task='binary')

model = ChurnModule()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCH = 150

model.train()
for i in range(1, EPOCH+1):
    running_loss = 0.0
    for features, labels in train_loader:
        model.train()
        optimizer.zero_grad()
        outputs = model(features)

        loss = criterion(outputs.squeeze(1), labels)
        f1_score.update(outputs.squeeze(1), labels)
        accuracy_score.update(outputs.squeeze(1), labels)
        recall_score.update(outputs.squeeze(1), labels)
        precision_score.update(outputs.squeeze(1), labels)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    accuracy = accuracy_score.compute().item()
    recall = recall_score.compute().item()
    precision = precision_score.compute().item()
    f1 = f1_score.compute().item()
    print(f'Train ----------> Epoch {i} -- Loss {(running_loss/len(train_loader)):.2f} -- Accuracy - {accuracy*100:.2f}% -- F1 - {f1*100:.2f}% --Recall - {recall*100:.2f}% -- Precision - {precision*100:.2f}%')


    model.eval()
    with torch.no_grad():
        for feature, label in val_loader:
            optimizer.zero_grad()
            output = model(feature)

            loss = criterion(output.squeeze(1), label)
            f1_score_.update(output.squeeze(1), label)
            accuracy_score_.update(output.squeeze(1), label)
            recall_score_.update(output.squeeze(1), label)
            precision_score_.update(output.squeeze(1), label)

            running_loss += loss.item()
            optimizer.step()

        accuracy = accuracy_score.compute().item()
        recall = recall_score.compute().item()
        precision = precision_score.compute().item()
        f1 = f1_score.compute().item()
    print(f'Validation -----> Loss {(running_loss/len(val_loader)):.2f} -- Accuracy - {accuracy*100:.2f}% -- F1 - {f1*100:.2f}% --Recall - {recall*100:.2f}% -- Precision - {precision*100:.2f}%')
torch.save(model.state_dict(), 'Iranian_Telecom_Churn_model.pth')