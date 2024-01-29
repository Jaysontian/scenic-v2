
def sigmoid(x):
    output = 1 / (1 + torch.exp(-x))
    return output


def cross_entropy_loss(p, y):
    #print(p.shape[0], y.shape[0])
    output = - (y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()
    return output

class LogisticRegression(LinearRegression):

    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__(input_size, output_size)

    def forward(self, x):
        output = super(LogisticRegression, self).forward(x)
        output = sigmoid(output)

        return output

    def get_loss(self, pred_logits, targets):
        new_targets = torch.nn.functional.one_hot(targets, num_classes=100)       # parameterize num_classes
        loss = cross_entropy_loss(pred_logits, new_targets)

        return loss

    def evaluate(self, data_loader):
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
            preds = preds.argmax(dim=1)
            correct += (preds == labels).sum().item()
            count += labels.shape[0]
            pbar.update()
        return correct / count
    
seed_everything(0)

logistic_model = LogisticRegression(3072, 100)
# hyperparam / learning rate opt
val_accs_1, losses_1 = logistic_model.fit(train_loader, val_loader, lr=1e-4)
val_accs_2, losses_2 = logistic_model.fit(train_loader, val_loader, lr=1e-2)