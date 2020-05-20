import torch
import torch.nn as nn

class MultiLayerNet(torch.nn.Module):
    def __init__(self, L, D_in, H, D_out):
        """
        In the constructor we instantiate multiple nn.Linear modules and assign them as
        member variables.
        """
        super(MultiLayerNet, self).__init__()
        self.linear_first = nn.Linear(D_in, H)
        self.linear_mids = nn.ModuleList([nn.Linear(H, H) for i in range(0, L - 2)])
        self.linear_last = nn.Linear(H, D_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.linear_first(x)
        x = self.relu(x)
        for layer in self.linear_mids:
            x = layer(x)
            x = self.relu(x)
        x = self.linear_last(x)
        return x