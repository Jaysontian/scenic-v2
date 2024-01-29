import torch
from tqdm import tqdm

# USING GPU CUDA

class LinearRegression(object):

    def __init__(self, input_size, output_size):
        self.linear_weights = torch.zeros((input_size, output_size), requires_grad=True)
        self.linear_bias = torch.zeros((output_size), requires_grad=True) # randomize bias (gradient=true used for backprop)
        self.linear_weights.data = self.linear_weights.to('cuda')
        self.linear_bias.data = self.linear_bias.to('cuda')

    def linear(self, x):
        output = torch.matmul(x, self.linear_weights) + self.linear_bias
        return output

    def forward(self, x):
        return self.linear(x).squeeze()

    def get_loss(self, preds, targets):
        loss = torch.mean((preds - targets)**2)
        return loss

    def fit(self, train_loader, val_loader, lr, epochs=5):
        # Fit the linear regression model to the training data using gradient descent
        # lr is the learning rate, epochs is the number of epochs

        # To store validation accuracy
        val_accs = []
        train_losses = []
        for epoch in range(epochs):
            # Create a progress bar using tqdm
            pbar = tqdm(total=len(train_loader),
                        desc=f'Epoch {epoch + 1}/{epochs}',
                        position=0,
                        leave=True)
            epoch_loss = 0
            iter = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                pred = self.forward(images)
                #print("SIZES: ", len(images), len(pred), len(labels))
                loss = self.get_loss(pred, labels)
                loss.backward()

                with torch.no_grad():
                    self.linear_weights.data -= lr * self.linear_weights.grad
                    self.linear_bias.data -= lr * self.linear_bias.grad

                self.linear_weights.grad.zero_()
                self.linear_bias.grad.zero_()

                pbar.set_postfix(loss=loss.item(), lr=lr)
                pbar.update()
                epoch_loss += loss.item()
                iter += 1
            # Calculate the validation accuracy
            val_acc = self.evaluate(val_loader)
            val_accs.append(val_acc)
            # Calculate the average training loss
            epoch_loss = epoch_loss / iter
            train_losses.append(epoch_loss)
            # Update the progress bar with the validation accuracy and training loss
            pbar.set_description(
                f'val_acc: {val_acc:.3f}, train_loss: {epoch_loss:.3f}')
        return val_accs, train_losses

    def evaluate(self, data_loader):
        # Evaluate the performance of the linear regression model on the dataset
        count = 0
        correct = 0
        pbar = tqdm(range(len(data_loader)),
                    desc='Evaluating:',
                    position=0,
                    leave=True)
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = self.forward(images)
            # Calculate the predicted labels
            preds = preds.round().int()
            correct += (preds == labels).sum().item()
            count += labels.shape[0]
            pbar.update()
        return correct / count
    
seed_everything(0)

linear_model = LinearRegression(3072, 1)
train_acc = linear_model.evaluate(train_loader)
val_acc = linear_model.evaluate(val_loader)
print('train accuracy:', train_acc)
print('val accuracy:', val_acc)

linear_model = LinearRegression(3072, 1)
val_accs_1, losses_1 = linear_model.fit(train_loader, val_loader, lr=1e-4)