def softmax(x):
    exp = torch.exp(x)
    output = exp / torch.sum(exp, dim=1, keepdim=True)
    return output


def nll_loss(p, y):
    probabilities = p[range(p.shape[0]), y]
    loss = -torch.log(probabilities)
    return loss

class SoftmaxRegression(LogisticRegression):

    def __init__(self, input_size, output_size):
        super(SoftmaxRegression, self).__init__(input_size, output_size)

    def forward(self, x):
        output = super(SoftmaxRegression, self).forward(x)
        output = softmax(output)
        return output

    def get_loss(self, pred_logits, targets):
        loss = nll_loss(pred_logits, targets).mean()
        return loss