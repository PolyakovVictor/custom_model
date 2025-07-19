import torch


class CustomNN:
    def __init__(self,):
        pass

    def ReLU(self, number: int) -> int: return number if number > 0 else 0



class CustomLinear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.randn(out_features, in_features, requires_grad=True)
        self.b = torch.randn(out_features)
    
    def forward(self, x):
        self.x = x
        output = self.W @ x + self.b
        return output

    def backward(self, dL_dy):
        dL_dW = dL_dy.view(-1, 1) @ self.x.view(1, -1) # out_features x in_features
        dL_db = dL_dy
        dL_dx = self.W.t() @ dL_dy
        return dL_dx, dL_dW, dL_db

if __name__ == '__main__':
    layer = CustomLinear(5, 3)
    x = torch.randn(5)
    y = layer.forward(x)
    dL_dy = torch.randn(3)

    dL_dx, dL_dW, dL_db = layer.backward(dL_dy)
    print('Gradient w.r.t input:', dL_dx)
    print('Gradient w.r.t weights:', dL_dW)
    print('Gradient w.r.t bias:', dL_db)