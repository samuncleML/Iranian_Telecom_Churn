from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from test import predictions
import pandas as pd

BASE_PATH = Path(__file__).resolve().parents[1]
y_test = pd.read_csv(BASE_PATH / 'data' / 'processed' / 'data_train.csv')['Churn']
model_prediction = predictions

accuracy = accuracy_score(y_test, model_prediction)
precision = precision_score(y_test, model_prediction)
recall = recall_score(y_test, model_prediction)
f1_score_ = f1_score(y_test, model_prediction)
auc_score = roc_auc_score(y_test, model_prediction)
roc_curve_ = roc_curve(y_test, model_prediction)
cm = confusion_matrix(y_test, model_prediction)
classificatio_summary = classification_report(y_test, model_prediction)

print(f"""
            ---Test---
      The accuracy is {accuracy*100:.2f}% on the test set
      The f1_score is {f1_score_*100:.2f}% on the test set
      The roc_auc is {auc_score*100:.2f}% on the test set
      The precision is {precision*100:.2f}% on the test set
      The recall is {recall*100:.2f}% on the test set
""")
print(classificatio_summary)

sns.heatmap(cm, annot=True)
plt.title('The Final Confusion Matrix on the Test Dataset')
plt.show()