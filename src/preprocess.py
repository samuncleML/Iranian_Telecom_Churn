import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split

BASE_PATH = Path(__file__).resolve().parents[1]
data = pd.read_csv(BASE_PATH / "data" / "raw" / "Customer Churn.csv")
# Keep all features including 'Status' and 'Complains'
data.head()

minmax = ['Tariff Plan', 'Status', 'Complains']
standard = ['Age Group', 'Age']
powerT = ['Call  Failure', 'Subscription  Length', 'Charge  Amount', 'Seconds of Use', 'Frequency of use', 
          'Frequency of SMS', 'Distinct Called Numbers', 'Customer Value']

st = StandardScaler()
pt = PowerTransformer()
mm = MinMaxScaler()

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_sample, y_train, y_sample = train_test_split(X, y, test_size=0.3, stratify=y, random_state=222)
X_test, X_val, y_test, y_val = train_test_split(X_sample, y_sample, test_size=0.5, stratify=y_sample, random_state=22)

def transform_tr(data, minmax, mm, standard, st, powerT, pt):
    """
    This function transforms data in a columns with the 
    specific distribution it follows.
    """
    data[minmax] = mm.fit_transform(data[minmax])
    data[standard] = st.fit_transform(data[standard])
    data[powerT] = pt.fit_transform(data[powerT])
    return data

def transform_tv(data, minmax, mm, standard, st, powerT, pt):
    """
    This function is the similar to *transform_tr* 
    only that that one is for train dataset and this 
    is for validation and testset.
    """
    data[minmax] = mm.transform(data[minmax])
    data[standard] = st.transform(data[standard])
    data[powerT] = pt.transform(data[powerT])
    return data

data_train = transform_tr(X_train, minmax, mm, standard, st, powerT, pt)
data_train['Churn'] = y_train
data_train.to_csv(BASE_PATH / 'data' / 'processed' / 'data_train.csv')

data_test = transform_tv(X_test, minmax, mm, standard, st, powerT, pt)
data_test['Churn'] = y_test
data_test.to_csv(BASE_PATH / 'data' / 'processed' / 'data_test.csv')

data_validation = transform_tv(X_val, minmax, mm, standard, st, powerT, pt)
data_validation['Churn'] = y_val
data_validation.to_csv(BASE_PATH / 'data' / 'processed' / 'data_validation.csv')