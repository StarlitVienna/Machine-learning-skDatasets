import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from NN import *
seed = 42
features, labels = make_blobs(n_samples=1000, n_features=2, random_state=720, centers=4)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=seed)

train_features = torch.from_numpy(train_features)
test_features = torch.from_numpy(test_features)
test_labels = torch.from_numpy(test_labels)
train_labels = torch.from_numpy(train_labels)

model = BlobModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
epochs = 100
train_model(model, optimizer, loss_fn, train_features, train_labels, epochs)

model.eval()
with torch.inference_mode():
    test_preds = torch.argmax(nn.Softmax(dim=1)(model(test_features)), dim=1)

figure, axis = plt.subplots(1, 2)

test_accuracy = torch.sum(torch.eq(test_preds, test_labels))/len(test_preds)
print(f"Accuracy --> {test_accuracy*100:.2f}%")

axis[0].scatter(test_features[:, 0], test_features[:, 1], c=test_preds, cmap=plt.cm.RdYlBu)
axis[0].set_title("NN")
axis[1].scatter(test_features[:, 0], test_features[:, 1], c=test_labels, cmap=plt.cm.RdYlBu)
axis[1].set_title("Data")
plt.show()
