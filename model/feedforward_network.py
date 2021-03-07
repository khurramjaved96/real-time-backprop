import torch
import torch.nn.functional as F
from torch import nn
import random
import numpy as np

class Column(nn.Module):
    def __init__(self, input_features, width, depth):
        super(Column, self).__init__()
        # self.linear_transformation = nn.Parameter(torch.ones(input_features, width))
        self.time_mod = 30 * depth + 2
        self.weights = nn.Parameter(torch.ones(width, width, depth))
        # torch.nn.init.uniform_(self.linear_transformation, -1*np.sqrt(1/width), np.sqrt(1/width))
        torch.nn.init.uniform_(self.weights, -1*np.sqrt(1/(width)), np.sqrt(1/width))
        self.state = torch.zeros(width, depth+1)
        self.output_weights = nn.Parameter(torch.ones(width,(depth+1)))
        torch.nn.init.uniform_(self.output_weights, -1 * np.sqrt(1 / (width+depth+1)), np.sqrt(1 /(width+depth+1)))
        self.gradients = torch.zeros(width, width, depth)
        self.stored_activations = torch.zeros(width, depth+1, self.time_mod)
        self.stored_gradiets = torch.zeros(width, depth)
        self.output_gradient = torch.zeros(width,depth+1, self.time_mod)
        self.output_weights_gradient = torch.zeros(width, depth +1)
        self.depth = depth
        self.time = 0


    def sparse_like(self, x, sparsity):
        return torch.bernoulli(torch.zeros_like(x) + sparsity)

    def zero_grad(self):
        self.grads = {}
        for named, param in self.named_parameters():
            self.grads[named] = torch.zeros_like(param.data)

    def forward(self, x):
        x_transformed = x
        # x_transformed = torch.sum(x.view(-1, 1)*self.linear_transformation, 0)

        self.state = torch.relu(torch.sum(self.weights*self.state[:, 0:self.depth].unsqueeze(1), 0))
        self.state = torch.cat((x_transformed.view(-1, 1), self.state), dim=1)
        # print(self.state)
        # self.state[:, 0] = x_transformed


        output = torch.sum(self.output_weights*self.state)
        # print(self.output_weights)
        with torch.no_grad():
            self.stored_activations[:, :, self.time] = self.state
            for a in range(0, self.depth+1):
                self.output_gradient[:, a, self.time] = self.output_weights[:, a]

            for a in range(0, self.depth):
                time_index = (self.time - self.depth*2 + (a+1)*2)%self.time_mod
                # print(self.time, time_index, a+1)
                if a == self.depth -1:
                    assert(time_index == self.time)
                    self.stored_gradiets[:, a] =  self.output_gradient[:, a+1, self.time]
                else:

                    relu_gradient = ((self.stored_activations[:, a+2, (time_index+1)%self.time_mod].unsqueeze(1)>0).float())
                    # print(a, relu_gradient)
                    dnext_dnow = self.stored_gradiets[:, a + 1].unsqueeze(0)

                    self.stored_gradiets[:, a] = torch.sum(self.weights[:, :, a+1]*relu_gradient * dnext_dnow, 1) + self.output_gradient[:, a+1, (time_index)%self.time_mod]

            self.output_weights_gradient = self.state.clone()

            for a in range(0, self.depth):
                time_index = (self.time - self.depth * 2 + (a + 1) * 2) % self.time_mod
                relu_grad  = ((self.stored_activations[:,a+1, (time_index)%self.time_mod]>0).float())
                self.gradients[:, :, a] = self.stored_activations[:,a, (time_index-1)%self.time_mod].unsqueeze(1)*(self.stored_gradiets[:, a]*(relu_grad).unsqueeze(0))



            self.time = (self.time + 1) % self.time_mod
        return output

    def step(self, lr, y):

        self.weights.data = self.weights + lr*torch.clamp((10 - y)*self.gradients, -10, 10)
        self.output_weights.data = self.output_weights + lr*torch.clamp((10 - y)*self.output_weights_gradient, -10, 10)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    set_seed(12)
    w= 100
    n = Column(w, width=w, depth=8)
    sum = 0
    avg_val = None

    for a in range(0, 10000000):

        x = torch.bernoulli(torch.zeros(w) + 0.5)


        y = n(x)
        # y.backward(retain_graph=True)
        # sum += y
        n.step(1e-4, y)
        if avg_val is None:
            avg_val = ((10 - y)**2).item()
        else:
            avg_val = avg_val*0.99 + 0.01*((10-y)**2).item()
        if a%100 == 0:
            print(avg_val)
    #
    # for nam, pa in n.named_parameters():
    #     if pa.grad is not None:
    #         if nam == "weights":
    #             print("real grad", pa.grad)
    #             print("real-time backpop", n.gradients)
    #             print(nam, pa.grad - n.gradients)
    #         else:
    #             print(n.output_weights_gradient - pa.grad)
    #


