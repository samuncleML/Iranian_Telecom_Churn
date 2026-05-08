from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from pathlib import Path

class ChurnDataSet(Dataset):
    def __init__(self, data):
        super(ChurnDataSet, self).__init__()
        self.data = data
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.iloc[:, 1:]
        self.data = torch.from_numpy(self.data.to_numpy()).float()
        #print(self.data.shape)
        self.data = self.data[:, :]
        #print(self.data.shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, -1]
        return x, y


    
#Data For Normal Training

BASE_PATH = Path(__file__).resolve().parents[1]
test_data = pd.read_csv(BASE_PATH / "data" / "processed" / "data_test.csv")
train_data = pd.read_csv(BASE_PATH / "data" / "processed" / "data_train.csv")
val_data = pd.read_csv(BASE_PATH / "data" / "processed" / "data_validation.csv")


test_set = ChurnDataSet(test_data)
train_set = ChurnDataSet(train_data)
val_set = ChurnDataSet(val_data)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=150, shuffle=False)
val_loader = DataLoader(val_set, batch_size=150, shuffle=False)


#Data For Cross-Validation

data_1 = pd.read_csv(BASE_PATH / 'data' / 'cross_val_data' / 'data_1.com')
data_2 = pd.read_csv(BASE_PATH / 'data' / 'cross_val_data' / 'data_2.com')
data_3 = pd.read_csv(BASE_PATH / 'data' / 'cross_val_data' / 'data_3.com')
data_4 = pd.read_csv(BASE_PATH / 'data' / 'cross_val_data' / 'data_4.com')

train_one = pd.concat([data_2, data_3, data_4]) ; val_one = data_1
train_two = pd.concat([data_1, data_3, data_4]) ; val_two = data_2
train_three = pd.concat([data_1, data_2, data_4]) ; val_three = data_3
train_four = pd.concat([data_1, data_2, data_3]) ; val_four = data_4

train_set1 = ChurnDataSet(train_one) ; val_set1 = ChurnDataSet(val_one)
train_set2 = ChurnDataSet(train_two) ; val_set2 = ChurnDataSet(val_two)
train_set3 = ChurnDataSet(train_three) ; val_set3 = ChurnDataSet(val_three)
train_set4 = ChurnDataSet(train_four) ; val_set4 = ChurnDataSet(val_four)

train_loader_one = DataLoader(train_set1, batch_size=32, shuffle=True) ; val_loader_one = DataLoader(val_set1, batch_size=75)
train_loader_two = DataLoader(train_set2, batch_size=32, shuffle=True) ; val_loader_two = DataLoader(val_set2, batch_size=75)
train_loader_three = DataLoader(train_set3, batch_size=32, shuffle=True) ; val_loader_three = DataLoader(val_set3, batch_size=75)
train_loader_four = DataLoader(train_set4, batch_size=32, shuffle=True) ; val_loader_four = DataLoader(val_set4, batch_size=75)

cross_val_train = [train_loader_one, train_loader_two, train_loader_three, train_loader_four]
cross_val_validation = [val_loader_one, val_loader_two, val_loader_three, val_loader_four]
