import torch
import numpy as np

class HW_normal(torch.nn.Module):
    def __init__(self, input_dim, output_dim, quantiles_per_dim=16, uniform_quantiles=False, focus_level=0.8, learning_rate_multiplier=1.0):
        super(HW_normal, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.quantiles_per_dim = quantiles_per_dim
        self.uniform_quantiles = uniform_quantiles
        self.focus_level = focus_level
        self.learning_rate_multiplier = learning_rate_multiplier

        percentile = np.linspace(0.0, 1.0, quantiles_per_dim * 2 + 1)[1::2]

        if uniform_quantiles:
            self.quantiles = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(percentile, dtype=torch.float32), requires_grad=False)])
            self.linear = torch.nn.ModuleList([torch.nn.Linear(quantiles_per_dim, output_dim)])
        else:
            self.quantiles = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(percentile), requires_grad=False) for _ in range(input_dim)])
            self.linear = torch.nn.ModuleList([torch.nn.Linear(quantiles_per_dim, output_dim) for _ in range(input_dim)])
    
    def forward(self, x):
        if x.dim() == 1:
            if self.input_dim != 1:
                raise ValueError("Input tensor x should have the same number of dimensions as input_dim")
            x = x.unsqueeze(1)
        else:
            shape = x.shape[-1]
            if shape != self.input_dim:
                raise ValueError("Input tensor x should have the same number of dimensions as input_dim")
        
        if self.uniform_quantiles:
            variance = torch.var(x.squeeze())
            x = x / torch.sqrt(variance)

            quantile = torch.quantile(x.squeeze(), self.quantiles[0], keepdim=False)
            x = quantile.unsqueeze(1)
            q = x.reshape(-1, self.quantiles_per_dim)

            d = torch.square(x - q)

            s = torch.nn.functional.softmax(d * -1.0, dim=-1)
            
            s_max, _ = torch.max(s, dim=-1, keepdim=True)
            d = d * 0.8 / s_max

            s = torch.nn.functional.softmax(d * -1.0, dim=-1)

            # x = x.unsqueeze(1)

            # shape = np.ones_like(list(x.shape))
            # shape[-1] = self.quantiles_per_dim
            # quantile = quantile.reshape(list(shape))

            # x = torch.square(x - quantile)
            # x = x * -1.0

            # x = torch.nn.functional.softmax(x, dim=-1)
            # x = x * self.focus_level
        else:
            pass
        
        return quantile, s

if __name__ == '__main__':
    net = HWnormal(1, 1, uniform_quantiles = True)

    test_x_np = np.random.randn(16 * 16).astype(np.float32)
    test_x_np = np.sort(test_x_np, axis=0)
    
    test_x = torch.tensor(test_x_np, dtype=torch.float32)  # Convert numpy array to torch tensor
    
    q, s = net(test_x)
    
    q_np = q.numpy().astype(np.float32)
    s_np = s.numpy().astype(np.float32)

    # for n in test_y_numpy:
    #     print(f"{n:.6f}", end=" ")
    # print("")

    for i, c in enumerate(zip(q_np, s_np)):
        print(f"{c[0]:.6f}", end=": ")
        for d in c[1]:
            print(f"{d:.6f}", end=" ")
        print("")

