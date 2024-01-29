import torch
import torch.nn as nn
import torch.nn.functional as F

## custom optimization implementations for SDG and Adams optimization for testing:

class SGD():
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.grad_buffer = {}

    def zero_grad(self):
        # Set the gradients of all parameters to zero.
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            velocity = self.grad_buffer.get(i, torch.zeros(param.grad.data.size()))
            velocity = velocity.to('cuda')
            param.data = param.data - self.lr * (velocity + param.grad)

class Adam():
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moments = {}
        self.second_moments = {}
        # Iteration counter
        self.t = 0

    def zero_grad(self):
        # Set the gradients of all parameters to zero.
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            # velocity = self.grad_buffer.get(i, torch.zeros(param.grad.data.size()))
            # velocity = velocity.to('cuda')
            # adam optimizer processing?

            g = param.grad.data
            self.first_moments[i] = self.beta1 * self.first_moments[i] + (1 - self.beta1) * g if i in self.first_moments else torch.zeros_like(param.data)
            self.second_moments[i] = self.beta2 * self.second_moments[i] + (1 - self.beta2) * (g ** 2) if i in self.second_moments else torch.zeros_like(param.data)
            m = self.first_moments[i] / (1 - self.beta1 ** self.t)
            v = self.second_moments[i] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m / (torch.sqrt(v) + self.eps)





class FastMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def train(model, train_loader, val_loader, optimizer, criterion, device,
          num_epochs):
    # Place model on device
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        # Use tqdm to display a progress bar during training
        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero out gradients
                optimizer.zero_grad()

                # Compute the logits and loss
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        # Evaluate the model on the validation set
        avg_loss, accuracy = evaluate(model, val_loader, criterion, device)
        print(
            f'Validation set: Average loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}'
        )


def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    # Compute the average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy


seed_everything(0)

model = FastMLP(input_size=3 * 32 * 32,
                hidden_size=1024,
                num_classes=len(miniplaces_train.label_dict))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train(model,
      train_loader,
      val_loader,
      optimizer,
      criterion,
      device,
      num_epochs=5)