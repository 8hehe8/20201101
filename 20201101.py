import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

from google.colab import drive

drive.mount("/content/drive")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_df = pd.read_csv("./drive/MyDrive/Colab Notebooks/gisthouseprice2021/price_data_tr.csv")
val_df = pd.read_csv("./drive/MyDrive/Colab Notebooks/gisthouseprice2021/price_data_val.csv")

corr_df = train_df.corr()["price"]
corr_df = corr_df.drop("price").copy()
# print(corr_df)
# print(corr_df.apply(lambda x: x>0.5 or x<-0.5))

# condition
cols = ["bathrooms", "sqft_living", "grade", "sqft_above", "sqft_living15"]

# train_x = train_df.drop(['id','price'], axis=1).copy()
train_x = train_df[cols]
train_x = train_x.select_dtypes(include="number")
train_y = train_df["price"]

# val_x = val_df.drop(['id','price'], axis=1)
val_x = val_df[cols]
val_x = val_x.select_dtypes(include="number")
val_y = val_df["price"]

col_numbers = train_x.shape[1]

# print(val_x)

tr_means, tr_maxs, tr_mins = dict(), dict(), dict()
val_means, val_maxs, val_mins = dict(), dict(), dict()

for col in cols:
    tr_means[col] = train_x[col].mean()
    tr_maxs[col] = train_x[col].max()
    tr_mins[col] = train_x[col].min()

    val_means[col] = val_x[col].mean()
    val_maxs[col] = val_x[col].max()
    val_mins[col] = val_x[col].min()

train_x = (train_x - train_x.mean()) / (train_x.max() - train_x.min())
val_x = (val_x - val_x.mean()) / (val_x.max() - val_x.min())

torch_x_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32)
torch_y_train = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1)

torch_x_val = torch.tensor(val_x.to_numpy(), dtype=torch.float32)
torch_y_val = torch.tensor(val_y, dtype=torch.float32).reshape(-1, 1)

# print(torch_x_train, torch_y_train)
# print(torch.sum(torch_x_train, dim=1))

train_data = TensorDataset(torch_x_train, torch_y_train)
print(train_data[0])
train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True)

H1, H2, H3 = 64, 32, 16
D_in = col_numbers
D_out = 1


class network(nn.Module):
    def __init__(self, D_in, H1, H2, H3, D_out):
        super(network, self).__init__()

        self.linear1 = nn.Linear(D_in, H1)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(H1, 1)
        # self.linear3 = nn.Linear(H2, H3)
        # self.linear4 = nn.Linear(H3, D_out)

    def forward(self, x):
        y_pred = self.linear1(x)
        y_pred = self.relu1(y_pred)
        y_pred = self.linear2(y_pred)
        # y_pred = self.linear2(y_pred).clamp(min=0)
        # y_pred = self.linear3(y_pred).clamp(min=0)
        # y_pred = self.linear4(y_pred)
        return y_pred


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal(m.weight, std=0.01)


model = network(D_in, H1, H2, H3, D_out)
model.to(device)


def train(num_epochs, train_dataloader):
    l = []
    for epoch in range(num_epochs):
        avg_loss = 0
        for idx, batch in enumerate(train_dataloader):
            input, target = batch[0], batch[1]
            input, target = Variable(input.to(device)), Variable(target.to(device))

            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item() * input.size(0)

        if epoch % 10 == 0:
            print(f"{epoch}, {avg_loss}")

        l.append(avg_loss / len(train_dataloader))
    return l


model.apply(init_weights)
print(model)
num_epochs = 90

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_func = nn.MSELoss(reduction="sum")
l = train(num_epochs, train_dataloader)

plt.figure()
plt.plot(range(num_epochs), l)
plt.show()


y_pred_train = model(torch_x_train.to(device))
y_pred_val = model(torch_x_val.to(device))

y_pred_train, y_pred_val = y_pred_train.detach().cpu().numpy(), y_pred_val.detach().cpu().numpy()

print(y_pred_train)
print(y_pred_val)

plt.figure()
plt.scatter(y_pred_train, train_y, c="blue", label="train data")
plt.scatter(y_pred_val, val_y, c="black", label="validation data")
plt.title("Linear regression")
plt.ylabel("Real Price")
plt.xlabel("Predicted Price")
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
plt.legend(loc="upper left")
plt.show()

print(y_pred_val[0:11])


y_val = model(torch_x_val[0:11, :].to(device))

print(y_val)

r2 = r2_score(val_y, y_pred_val)
print(r2)


test_df = pd.read_csv("./drive/MyDrive/Colab Notebooks/gisthouseprice2021/price_data_ts.csv")

# train_x = train_df.drop(['id','price'], axis=1).copy()
test_x = test_df[cols]
test_x = test_x.select_dtypes(include="number")

test_means, test_maxs, test_mins = dict(), dict(), dict()

for col in cols:
    test_means[col] = test_x[col].mean()
    test_maxs[col] = test_x[col].max()
    test_mins[col] = test_x[col].min()

test_x = (test_x - test_x.mean()) / (test_x.max() - test_x.min())

torch_x_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32)

y_pred_test = model(torch_x_test.to(device))
print(y_pred_test)


new_id = np.array([str(c).zfill(10) for c in test_df.loc[:, "id"]])
# print(new_id)
date = np.array(test_df["date"])
# print(date)

# print(new_id.shape, date.shape)
# print(type(new_id[0]),type(date[0]))
id_plus_date = [a + b for a, b in zip(new_id, date)]
# print(id_plus_date)

test_df["id"] = id_plus_date
test_df["price"] = y_pred_test.detach().cpu().numpy()

test_df = test_df[["id", "price"]]


test_df.to_csv("./test_pred2.csv", index=False)

