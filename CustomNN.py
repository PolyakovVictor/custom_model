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
        output = self.W @ x + self.b
        return output

    def backward(self,):
        pass


if __name__ == '__main__':
    l1 = CustomLinear(5, 3)
    print('Linear bias', l1.b)
    print('Linear W', l1.W)
    test = torch.randn(5)
    print(f'test: {test}')
    l1.forward(test)