import torchvision.transforms as transforms
import numpy as np

from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


transform = transforms.ToTensor()
train_dataset = FashionMNIST(root='../data', train=True, transform=transform)
test_dataset = FashionMNIST(root='../data', train=False, transform=transform)

# subset = torch.utils.data.Subset(train_dataset, range(1000))
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def cross_entropy_loss(y_pred, y_true):
    exps = np.exp(y_pred-np.max(y_pred, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    loss =  -np.sum(y_true * np.log(probs + 1e-9)) / y_true.shape[0]

    return loss, probs

def cross_entropy_grad(probs, y_true):
    return (probs - y_true) / y_true.shape[0]

def softmax(x, axis=1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
    

class CustomReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dL_dy):
        return dL_dy * self.mask
    

class CustomMLP:
    def __init__(self,):
        self.fc1 = CustomLinear(784, 50)
        self.relu = CustomReLU()
        self.fc2 = CustomLinear(50, 10)

    def forward(self, x): # x: (batch_size, 784)
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
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        # x = x.view(-1, 28)
        self.x = x
        return x @ self.W + self.b

    def backward(self, dL_dy):
        self.dW = self.x.T @ dL_dy # out_features x in_features
        self.db = dL_dy.sum(axis=0)
        # dL_dx = self.W.T @ dL_dy
        return dL_dy @ self.W.T
    
    def step(self, learning_rate: float = 0.0001):
        w = self.W - learning_rate * self.dW
        b = self.b - learning_rate * self.db
        self.W = w
        self.b = b

if __name__ == '__main__':
    model = CustomMLP()
    learning_rate = 0.001

    losses = []

    for epoch in range(10):
        for batch, (images, labels) in enumerate(train_loader):
            images = images.numpy()
            labels = labels.numpy()
            images = images.reshape(images.shape[0], -1)
            # images = images.reshape(-1, 784)[0]
            y_pred = model.forward(images)
            y_true = np.zeros((images.shape[0],10))
            for i,v in enumerate(labels):
                y_true[i][v] = 1
            # sq_loss = (y_pred - y_true) ** 2
            # loss = np.mean(sq_loss) # .mean arithmetic mean 
            loss, probs = cross_entropy_loss(y_pred=y_pred, y_true=y_true)
            losses.append(loss)
            
            dL_dy = cross_entropy_grad(probs=probs, y_true=y_true)

            model.backward(dL_dy)
            model.step(learning_rate)

            if batch % 1000 == 0:
                print(f"Epoch {epoch}: loss = {loss}")
    len_test_dataset = len(test_loader)
    correct_result = 0
    for batch, (images, labels) in enumerate(test_loader):
        images = images.numpy()
        labels = labels.numpy()
        images = images.reshape(-1, 784)[0]
        y_pred = model.forward(images)
        probs = softmax(y_pred[np.newaxis, :], axis=1)[0]
        predicted_class = np.argmax(probs)
        y_true = np.zeros(10)
        y_true[labels.item()] = 1
        if predicted_class == labels.item():
            correct_result += 1
        print(f'Prediction probabilities: {probs}')
        print(f'Predicted class: {predicted_class}')
        print(f'Real class: {labels.item()}')
    print(f'Correct result: {correct_result}')