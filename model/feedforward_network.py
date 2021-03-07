import random

import numpy as np
import torch
from torch import nn


class Realtime_FFN(nn.Module):
    def __init__(self, input_features, width, depth):
        super(Realtime_FFN, self).__init__()

        self.time_mod = 30 * depth + 2
        self.weights = nn.Parameter(torch.ones(width, width, depth))
        torch.nn.init.uniform_(self.weights, -1 * np.sqrt(1 / (width)), np.sqrt(1 / width))
        self.state = torch.zeros(width, depth + 1)
        self.output_weights = nn.Parameter(torch.ones(width, (depth + 1)))
        torch.nn.init.uniform_(self.output_weights, -1 * np.sqrt(1 / (width + depth + 1)),
                               np.sqrt(1 / (width + depth + 1)))
        self.gradients = torch.zeros(width, width, depth)
        self.stored_activations = torch.zeros(width, depth + 1, self.time_mod)
        self.stored_gradiets = torch.zeros(width, depth)
        self.output_gradient = torch.zeros(width, depth + 1, self.time_mod)
        self.output_weights_gradient = torch.zeros(width, depth + 1)
        self.depth = depth
        self.time = 0

    def forward(self, x):

        self.state = torch.relu(torch.sum(self.weights * self.state[:, 0:self.depth].unsqueeze(1), 0))
        self.state = torch.cat((x.view(-1, 1), self.state), dim=1)
        output = torch.sum(self.output_weights * self.state)

        return output

    def accumulate_gradient(self):
        # Real-time backward pass

        # To do: Use views to implement it without looping from 0 to depth.
        with torch.no_grad():
            self.stored_activations[:, :, self.time] = self.state
            for a in range(0, self.depth + 1):
                self.output_gradient[:, a, self.time] = self.output_weights[:, a]

            for a in range(0, self.depth):
                time_index = (self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod

                if a == self.depth - 1:
                    self.stored_gradiets[:, a] = self.output_gradient[:, a + 1, self.time]
                else:

                    relu_gradient = (
                        (self.stored_activations[:, a + 2, (time_index + 1) % self.time_mod].unsqueeze(1) > 0).float())
                    dnext_dnow = self.stored_gradiets[:, a + 1].unsqueeze(0)
                    gradient_from_later_layers = torch.sum(self.weights[:, :, a + 1] * relu_gradient * dnext_dnow, 1)
                    gradient_from_output = self.output_gradient[:, a + 1, (time_index) % self.time_mod]
                    self.stored_gradiets[:, a] = gradient_from_later_layers + gradient_from_output

            self.output_weights_gradient += self.state.clone()

            for a in range(0, self.depth):
                time_index = (self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod
                relu_grad = ((self.stored_activations[:, a + 1, (time_index) % self.time_mod] > 0).float())
                self.gradients[:, :, a] += self.stored_activations[:, a, (time_index - 1) % self.time_mod].unsqueeze(
                    1) * (self.stored_gradiets[:, a] * (relu_grad).unsqueeze(0))

            self.time = (self.time + 1) % self.time_mod

    def zero_grad(self):
        self.output_weights_gradient *= 0
        self.gradients *= 0

    def step(self, lr, y):

        self.weights.data = self.weights + lr * (10 - y) * self.gradients
        self.output_weights.data = self.output_weights + lr * (10 - y) * self.output_weights_gradient


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    set_seed(12)
    w = 100
    n = Realtime_FFN(w, width=w, depth=8)
    sum = 0
    avg_val = None

    for a in range(0, 10000000):

        x = torch.bernoulli(torch.zeros(w) + 0.5)
        y = n(x)
        n.accumulate_gradient()
        n.step(1e-4, y)
        n.zero_grad()
        if avg_val is None:
            avg_val = ((10 - y) ** 2).item()
        else:
            avg_val = avg_val * 0.99 + 0.01 * ((10 - y) ** 2).item()
        if a % 100 == 0:
            print(avg_val)
