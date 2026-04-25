from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch

class ChurnDataSet(Dataset):
    def __init__(self, path):
        super(ChurnDataSet, self).__init__()
        self.data = pd.read_csv(path).drop(['Unnamed: 0'], axis=1)
        self.data = torch.from_numpy(self.data.to_numpy()).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, -1]
        return x, y

test_set = ChurnDataSet(r"C:\Users\Administrator\Documents\Data_Science__Projects\Iranian_Telecom_Churn\data\processed\data_test.csv")
train_set = ChurnDataSet(r"C:\Users\Administrator\Documents\Data_Science__Projects\Iranian_Telecom_Churn\data\processed\data_train.csv")
val_set = ChurnDataSet(r"C:\Users\Administrator\Documents\Data_Science__Projects\Iranian_Telecom_Churn\data\processed\data_validation.csv")

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=150, shuffle=False)
val_loader = DataLoader(val_set, batch_size=150, shuffle=False)