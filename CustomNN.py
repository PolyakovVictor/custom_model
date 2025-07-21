import torch

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


torch.set_grad_enabled(False)

transform = transforms.ToTensor()
train_dataset = FashionMNIST(root='../data', train=True, transform=transform)
test_dataset = FashionMNIST(root='../data', train=False, transform=transform)

# subset = torch.utils.data.Subset(train_dataset, range(1000))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class CustomReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dL_dy):
        return dL_dy * self.mask.float()
    

class CustomMLP:
    def __init__(self,):
        self.fc1 = CustomLinear(784, 50)
        self.relu = CustomReLU()
        self.fc2 = CustomLinear(50, 10)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        return x
    
    def backward(self, dL_dt):
        d2 = self.fc2.backward(dL_dt)
        d1 = self.relu.backward(d2)
        self.fc1.backward(d1)
    
    def step(self, lr):
        self.fc1.step(lr)
        self.fc2.step(lr)



class CustomLinear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.randn(out_features, in_features)
        self.b = torch.randn(out_features)

        self.dW = torch.zeros_like(self.W)
        self.db = torch.zeros_like(self.b)

    def forward(self, x):
        # x = x.view(-1, 28)
        self.x = x
        output = self.W @ x.view(-1) + self.b
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
    model = CustomMLP()
    learning_rate = 0.0001

    x = torch.randn(5)
    losses = []

    for epoch in range(10):
        for batch, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 784)[0]
            y_pred = model.forward(images)
            y_true = torch.zeros(10)
            y_true[labels.item()] = 1
            loss = torch.mean((y_pred - y_true) ** 2) # .mean arithmetic mean 
            losses.append(loss.item())
            
            dL_dy = 2 * (y_pred - y_true) / y_pred.numel()

            model.backward(dL_dy)
            with torch.no_grad():
                model.step(learning_rate)

            if batch % 1000 == 0:
                print(f"Epoch {epoch}: loss = {loss.item()}")
    len_test_dataset = len(test_loader)
    correct_result = 0
    for batch, (images, labels) in enumerate(test_loader):
        images = images.view(-1, 784)[0]
        y_pred = model.forward(images)
        probs = torch.softmax(y_pred, dim=0)
        predicted_class = torch.argmax(probs).item()
        y_true = torch.zeros(10)
        y_true[labels.item()] = 1
        loss = torch.mean((y_pred - y_true) ** 2) 
        losses.append(loss.item())
        if predicted_class == labels.item():
            correct_result += 1
        print(f'Prediction probabilities: {probs}')
        print(f'Predicted class: {predicted_class}')
        print(f'Real class: {labels.item()}')
    print(f'Currect result: {correct_result}')