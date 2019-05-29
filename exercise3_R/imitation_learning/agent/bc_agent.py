import torch
from agent.networks import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"

class BCAgent:

    def __init__(self, lr=0.001, history_length=1, optimizer=torch.optim.Adam):
        # TODO: Define network, loss function, optimizer
        self.net = CNN(history_length=history_length, n_classes=5)
        self.net = self.net.to(device)
        self.loss_criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer(self.net.parameters(), lr=lr)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        X_batch = torch.tensor(X_batch).to(device, dtype=torch.float)
        y_batch = torch.tensor(y_batch).to(device, dtype=torch.long)

        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        # TODO: forward + backward + optimize
        outputs = self.net(X_batch)
        loss = self.loss_criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        return outputs, loss

    def predict(self, X, y=None):
        # TODO: forward pass
        with torch.no_grad():
            X = torch.tensor(X).to(device, dtype=torch.float)
            outputs = self.net(X)
            loss = None
            if y is not None:
                y = torch.tensor(y).to(device, dtype=torch.long)
                loss = self.loss_criterion(outputs, y).item()
        return outputs, loss

    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))
