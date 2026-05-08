# ROC Curve and Confusion Matrix

The following figure illustrates the model's performance on the test set using two key evaluation metrics:

- **ROC Curve (Receiver Operating Characteristic):**
  - The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
  - The Area Under the Curve (AUC) quantifies the model's ability to distinguish between churn and non-churn customers. An AUC close to 1.0 indicates excellent discrimination.

- **Confusion Matrix:**
  - The confusion matrix provides a summary of prediction results, showing the counts of true positives, true negatives, false positives, and false negatives.
  - This helps visualize the model's accuracy and the types of errors it makes.

<img width="1954" height="975" alt="image" src="https://github.com/user-attachments/assets/7c04cf31-594c-442e-9257-3dba129473ae" />

In this project, the model achieves a high AUC and demonstrates strong classification performance, as seen in the confusion matrix. This indicates the model is effective at identifying customers who are likely to churn.
