import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[5], output_dim=2):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(model, X, y, epochs=1000, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == y).float().mean().item()
            print(f"Epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}  acc={acc:.4f}")

    return model