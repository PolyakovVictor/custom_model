import torch


class CustomNN:
    def __init__(self,):
        pass

    def ReLU(self, number: int) -> int: return number if number > 0 else 0



class CustomLinear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.randn(out_features, in_features)
        self.b = torch.randn(out_features)

        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)
    def forward(self, x):
        self.x = x
        output = self.W @ x + self.b
        return output

    def backward(self, dL_dy):
        self.dW = dL_dy.view(-1, 1) @ self.x.view(1, -1) # out_features x in_features
        self.db = dL_dy
        dL_dx = self.W.t() @ dL_dy
        return dL_dx
    
    def step(self, learning_rate: float = 0.0001):
        w = self.W - learning_rate * self.dW
        b = self.b - learning_rate * self.db
        self.W = w
        self.b = b

if __name__ == '__main__':
    torch.manual_seed(0)
    layer = CustomLinear(5, 3)
    learning_rate = 0.0001

    x = torch.randn(5)
    y_true  = torch.tensor([1.0, 0.0, -1.0])
    for epoch in range(100):

        y_pred = layer.forward(x)

        loss = torch.mean((y_pred - y_true) ** 2)
        
        dL_dy = 2 * (y_pred - y_true) / y_pred.numel()

        layer.backward(dL_dy)
        layer.step(learning_rate=learning_rate)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")