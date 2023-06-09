import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from NN import *

device = "cuda" if torch.cuda.is_available() else "cpu"

n_samples=5000
train_size = 0.8
features, labels = make_circles(n_samples, noise=0, random_state=8)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=(1-train_size), random_state=8)

"""
model = nn.Sequential(
        nn.Linear(in_features=2, out_features=8*8, dtype=torch.double),
        nn.ReLU(),
        nn.Linear(in_features=8*8, out_features=8*8*8, dtype=torch.double),
        nn.ReLU(),
        nn.Linear(in_features=8*8*8, out_features=1, dtype=torch.double),
        )
"""
model = NNModel()
model.to(device)
epochs = 1000
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.BCEWithLogitsLoss()

train_features = torch.from_numpy(train_features).double()
train_labels = torch.from_numpy(train_labels).double()
test_features = torch.from_numpy(test_features).double()
test_labels = torch.from_numpy(test_labels).double()

train_model(model, loss_fn, optimizer, train_features, train_labels, epochs)

preds = make_preds(model, test_features)

accuracy = torch.sum(torch.eq(preds, test_labels.unsqueeze(dim=1)))/(len(preds))
#print(test_labels.unsqueeze(dim=1))
print(f"Test accuracy --> {accuracy*100}%")

figure, axis = plt.subplots(1, 2)
axis[0].scatter(test_features[:, 0], test_features[:, 1], c=preds, cmap=plt.cm.RdYlBu)
axis[0].set_title("NN")
axis[1].scatter(test_features[:, 0], test_features[:, 1], c=test_labels, cmap=plt.cm.RdYlBu)
axis[1].set_title("Data")
plt.show()
