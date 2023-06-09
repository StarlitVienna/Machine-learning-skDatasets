import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from NN import *

n_samples = 100
noise = 0.01
seed = 8
features, labels = make_circles(n_samples=n_samples, noise=noise, random_state=seed)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=seed)

train_features = torch.from_numpy(train_features).type(torch.float32)
test_features = torch.from_numpy(test_features).type(torch.float32)
train_labels = torch.from_numpy(train_labels).type(torch.float32)
test_labels = torch.from_numpy(test_labels).type(torch.float32)

model0 = Model()
optimizer = torch.optim.SGD(model0.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

epochs = 1000
train_model(model0, optimizer, loss_fn, train_features, train_labels, epochs)

model0.eval()
with torch.inference_mode():
    preds = torch.argmax(nn.Softmax(dim=1)(model0(test_features)), dim=1)
print(preds)


figure, axis = plt.subplots(1, 2)
axis[0].set_title("NN")
axis[0].scatter(test_features[:, 0], test_features[:, 1], c=preds, cmap=plt.cm.RdYlBu)
axis[1].scatter(test_features[:, 0], test_features[:, 1], c=test_labels, cmap=plt.cm.RdYlBu)
axis[1].set_title("Test data")
plt.show()
