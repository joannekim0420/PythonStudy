import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MultilayerPerceptron, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """args:
        x_in (torch.Tensor): input data tensor
        x_in.shape -> (batch, input_dim)
        apply_softmax : softmax activation flag
        """

        layer1 = F.relu(self.fc1(x_in))
        out_layer = self.fc2(layer1)

        if apply_softmax:
            output = F.softmax(out_layer)
        return output

if __name__ == "__main__":
    batch_size = 2
    input_dim = 3
    hidden_dim = 100
    output_dim = 4

    mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)

    print(mlp)