import config
import torch
import torch.nn as nn

class MODEL_NAME(nn.Module):
    def __init__(self):
        super(MODEL_NAME, self).__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    from random import choice
    batches = [2, 4, 8]
    model = MODEL_NAME()
    x = torch.randn(2, 1, 32, 128)
    print('input', x.size())
    out = model(x)
    print(out.size())

