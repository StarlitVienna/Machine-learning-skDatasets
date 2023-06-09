import torch
from torch import nn

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.layer_0 = nn.Linear(in_features=2, out_features=8*8, dtype=torch.float64)
        self.layer_1 = nn.Linear(in_features=8*8, out_features=8*8*8, dtype=torch.float64)
        self.layer_2 = nn.Linear(in_features=8*8*8, out_features=1, dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_0 = self.act_fn(self.layer_0(x))
        out_1 = self.act_fn(self.layer_1(out_0))
        out_2 = self.layer_2(out_1)
        return out_2

def train_model(model, loss_fn, optimizer, train_data, train_labels, epochs):
    model.train()
    for epoch in range(epochs):
        preds = model(train_data).squeeze()
        loss = loss_fn(preds, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()

def make_preds(model, data):
    model.eval()
    with torch.inference_mode():
        preds = torch.round(torch.sigmoid(model(data)))
    return preds

