import torch
import numpy as np
import convolution_emulator


KERNEL_SIZE = 3

class OwnConv2d:
    def __init__(self, input_size: int = 28, kernel_size: int = 3, ):
        self._input_size = input_size
        self._kernel_size = kernel_size
        self._output_size = (self._input_size - self._kernel_size + 1)
        
    
    def calc(self, tensor, kernel):
        output=torch.zeros(self._output_size, self._output_size)
        for x in range(self._output_size):
            for y in range(self._output_size):
                sum = 0.0
                for kx in range(self._kernel_size):
                    for ky in range(self._kernel_size):
                        input_x = x + kx
                        input_y = y + ky
                        if input_x < self._input_size and input_y < self._input_size:
                            sum += tensor[input_x][input_y] * kernel[kx][ky]
                output_position = x * self._output_size + y
                output[output_position // self._output_size][output_position % self._output_size] = float(sum)
        return output


# Prepare test data (e.g., MNIST-like 28x28 image)
input_image = torch.randn(28, 28).numpy()  # Random grayscale image
kernel = torch.ones(KERNEL_SIZE, KERNEL_SIZE).numpy() / 9.0    # Simple averaging kernel
print(f'My input_image: {input_image}')

# Run emulator
own_conv = OwnConv2d(kernel_size=KERNEL_SIZE)
python_result = own_conv.calc(input_image, kernel).squeeze().detach().numpy()
result = convolution_emulator.convolve(input_image, kernel)


# Compare with PyTorch Conv2d
conv = torch.nn.Conv2d(1, 1, kernel_size=KERNEL_SIZE, stride=1, padding=0, bias=False)
conv.weight.data = torch.tensor(kernel).reshape(1, 1, KERNEL_SIZE, KERNEL_SIZE)
input_tensor = torch.tensor(input_image).reshape(1, 1, 28, 28)
pytorch_result = conv(input_tensor).squeeze().detach().numpy()

# Verify results
# print("Emulator output shape:", result.sh/ape)
print("Python Emulator output shape:", python_result.shape)
print("PyTorch output shape:", pytorch_result.shape)
print("Max difference pytoch vs custom python:", np.max(np.abs(python_result - pytorch_result)))
print("Max difference pytoch vs custom C++:", np.max(np.abs(result - pytorch_result)))
print("Max difference python vs custom C++:", np.max(np.abs(result - python_result)))
