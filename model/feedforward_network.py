import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

class Realtime_FFN(nn.Module):
    def __init__(self, width, depth, device):
        super(Realtime_FFN, self).__init__()

        self.time_mod = 3 * depth + 2
        self.weights = nn.Parameter(torch.ones(width, width, depth))
        torch.nn.init.uniform_(self.weights, -1 * np.sqrt(1 / (width)), np.sqrt(1 / width))
        # torch.nn.init.uniform_(self.weights, 0.01, 1)
        self.state = torch.zeros(width, depth + 1).to(device)
        self.output_weights = nn.Parameter(torch.ones(width, (depth + 1)))
        torch.nn.init.uniform_(self.output_weights, -1 * np.sqrt(1 / (width + depth + 1)),
                               np.sqrt(1 / (width + depth + 1)))
        torch.nn.init.zeros_(self.output_weights)
        self.gradients = torch.zeros(width, width, depth).to(device)
        self.stored_activations = torch.zeros(width, depth + 1, self.time_mod).to(device)
        self.stored_gradiets = torch.zeros(width, depth).to(device)
        self.output_gradient = torch.zeros(width, depth + 1, self.time_mod).to(device)
        self.output_weights_gradient = torch.zeros(width, depth + 1).to(device)
        self.depth = depth
        self.time = 0

    def forward(self, x):

        self.state = torch.relu(torch.sum(self.weights * self.state[:, 0:self.depth].unsqueeze(1), 0))
        self.state = torch.cat((x.view(-1, 1), self.state), dim=1)
        output = torch.sum(self.output_weights * self.state)

        return output

    def accumulate_gradient(self, prediction, target):
        # Real-time backward pass

        error = (target - prediction)*-1

        with torch.no_grad():
            # Gradient of error w.r.t output feature
            self.stored_activations[:, :, self.time] = self.state
            self.output_gradient[:, :, self.time] = self.output_weights*error

            # Gradient of error w.r.t all features
            time_indices = np.array([(self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod for a in list(range(0, self.depth-1))])
            base_indexing = np.array(list(range(0, self.depth-1 )))
            relu_gradient = ((self.stored_activations[:, base_indexing + 2, (time_indices + 1) % self.time_mod].unsqueeze(1) > 0).float())
            dnext_dnow = self.stored_gradiets[:, base_indexing + 1].unsqueeze(0)

            gradient_from_later_layers = torch.sum(self.weights[:, :, 1: self.depth] * relu_gradient * dnext_dnow,1)
            gradient_from_output = self.output_gradient[:, base_indexing + 1, time_indices]
            self.stored_gradiets[:, 0:self.depth -1] = gradient_from_later_layers + gradient_from_output

            self.stored_gradiets[:, self.depth - 1] = self.output_gradient[:, self.depth, self.time]

            if self.output_weights.grad is None:
                self.output_weights.grad = self.state.clone()*error
            else:
                self.output_weights.grad += self.state.clone()*error


            # Gradient of error w.r.t parameters
            time_indices = np.array([(self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod for a in list(range(0, self.depth))])


            normal_list =  np.array(list(range(0, self.depth)))
            relu_grad = ((self.stored_activations[:, normal_list + 1, time_indices] > 0).float())
            if self.weights.grad is None:
                self.weights.grad = self.stored_activations[:, normal_list, (time_indices - 1) % self.time_mod].unsqueeze(
                    1) * (self.stored_gradiets[:, normal_list] * (relu_grad).unsqueeze(0))
            else:
                self.weights.grad += self.stored_activations[:, normal_list,
                                                     (time_indices - 1) % self.time_mod].unsqueeze(
                    1) * (self.stored_gradiets[:, normal_list] * (relu_grad).unsqueeze(0))

            self.time = (self.time + 1) % self.time_mod

    def accumulate_gradient_loopy(self, prediction, target):
        # Real-time backward pass

        # To do: Use views to implement it without looping from 0 to depth.
        error = (target - prediction)*-1
        with torch.no_grad():
            self.stored_activations[:, :, self.time] = self.state
            self.output_gradient[:, :, self.time] = self.output_weights*error
            time_indices = np.array([(self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod for a in list(range(0, self.depth-1))])
            # self.stored_gradiets[:, self.depth - 1] = self.output_gradient[:, self.depth, self.time]
            # print(time_indices)



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


            self.output_weights_gradient += self.state.clone()*error

            time_indices = np.array([(self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod for a in list(range(0, self.depth))])
            for a in range(0, self.depth):
                time_index = (self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod
                relu_grad = ((self.stored_activations[:, a + 1, (time_index) % self.time_mod] > 0).float())
                self.gradients[:, :, a] += self.stored_activations[:, a, (time_index - 1) % self.time_mod].unsqueeze(
                    1) * (self.stored_gradiets[:, a] * (relu_grad).unsqueeze(0))

            self.time = (self.time + 1) % self.time_mod


    #
    # def step(self, lr):
    #
    #     self.weights.data = self.weights - lr * self.gradients
    #     self.output_weights.data = self.output_weights - lr  * self.output_weights_gradient


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":

    gpu_to_use = 0
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(gpu_to_use))
        # logger.info("Using gpu : %s", 'cuda:' + str(gpu_to_use))
    else:
        device = torch.device('cpu')


    set_seed(12)
    w = 50

    n = Realtime_FFN( width=w, depth=50, device=device).to(device)
    sum = 0
    avg_val = None
    from timeit import default_timer as timer

    opti = Adam(n.parameters(), 1e-4, (0.9, 0.999))

    start = timer()

    for a in range(0, 200000000):
        with torch.no_grad():
            if a < 10 or True:
                x = torch.bernoulli(torch.zeros(w) + 0.5).to(device)
            else:
                x = torch.bernoulli(torch.zeros(w) + 0).to(device)
            y = n(x)
            # y.backward(retain_graph=True)
            opti.zero_grad()
            n.accumulate_gradient(y, 10)
            opti.step()

            if avg_val is None:
                avg_val = ((10 - y) ** 2).item()
            else:
                avg_val = avg_val * 0.99 + 0.01 * ((10 - y) ** 2).item()
            if a % 100 == 99:
                print(1000*((timer() - start)/a), "ms")
                # print(a, time() - start)
                print(avg_val)
#
    # for named, param in n.named_parameters():
    #     if named == "weights":
    #         print("GT", param.grad)
    #         print("Realtime", n.gradients)
    #         print(torch.max(torch.abs(param.grad - n.gradients)))
    #     else:
    #         # print(param.grad)
    #         print(n.output_weights_gradient - param.grad)

