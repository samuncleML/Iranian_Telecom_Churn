import torch
from torchmetrics import F1Score, Accuracy, Recall, Precision
from dataset import test_loader
from module import ChurnModule

f1_score = F1Score(task='binary')
accuracy_score = Accuracy(task='binary')
recall_score = Recall(task='binary')
precision_score = Precision(task='binary')

model = ChurnModule()
model.load_state_dict(torch.load(r'C:\Users\Administrator\Documents\Data_Science__Projects\Iranian_Telecom_Churn\models\Iranian_Telecom_Churn_model.pth'))
predictions = []
model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        outputs = model.forward(features)

        f1_score.update(outputs.squeeze(1), labels)
        accuracy_score.update(outputs.squeeze(1), labels)
        recall_score.update(outputs.squeeze(1), labels)
        precision_score.update(outputs.squeeze(1), labels)
        outs = [round(float(i), 2) for i in list(outputs.squeeze(1))]
        for i in outs:
            predictions.append(i)

    
    accuracy = accuracy_score.compute().item()
    recall = recall_score.compute().item()
    precision = precision_score.compute().item()
    f1 = f1_score.compute().item()

    print(f'Test -----> Accuracy - {accuracy*100:.2f}% -- F1 - {f1*100:.2f}% --Recall - {recall*100:.2f}% -- Precision - {precision*100:.2f}%')