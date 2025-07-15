import torch
import numpy as np
import convolution_emulator
import torch.nn as nn


KERNEL_SIZE = 3
IN_CHANNELS = 3
BATCH_SIZE = 1
OUT_CHANNELS = 3

class OwnConv2d:
    def __init__(self, input_size: int = 28, kernel_size: int = 3, in_channels: int = 1, out_channels: int = 1 ):
        self._input_size = input_size
        self._kernel_size = kernel_size
        self._output_size = (self._input_size - self._kernel_size + 1)
        self._in_channels = in_channels
        self._out_channels = out_channels
        
    
    def calc(self, tensor: np.ndarray, kernel: np.ndarray):
        output=torch.zeros(self._out_channels, self._output_size, self._output_size)
        for out_channel in range(self._out_channels):
            for in_channel in range(self._in_channels):
                for x in range(self._output_size):
                    for y in range(self._output_size):
                        sum = 0.0
                        for kx in range(self._kernel_size):
                            for ky in range(self._kernel_size):
                                input_x = x + kx
                                input_y = y + ky
                                if input_x < self._input_size and input_y < self._input_size:

                                    sum += tensor[0][in_channel][input_x][input_y] * kernel[out_channel][in_channel][kx][ky]
                        output[out_channel][x][y] += float(sum)
        return output


    def vector_calc(self, tensor: np.ndarray, kernel: np.ndarray):
        tensor = tensor.astype(np.float32)
        
        # (B, C_in, H, W) → (B, C_in, H_out, W_out, K, K)
        patches = np.lib.stride_tricks.sliding_window_view(
            tensor, 
            (self._kernel_size, self._kernel_size), 
            axis=(2, 3)
        )
        
        # einsum: b - batch, c - in_channels, o - out_channels, x/y - spatial, i/j - kernel
        output = np.einsum('bcxyij,ocij->boxy', patches, kernel)
        return output.squeeze()





# Prepare test data (e.g., MNIST-like 28x28 image)
input_image = torch.randn(BATCH_SIZE, IN_CHANNELS, 28, 28).numpy()  # Random grayscale image
kernel = torch.ones(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).numpy() / 9.0    # Simple averaging kernel
# print(f'My input_image: {input_image}')

# Run emulator
own_conv = OwnConv2d(in_channels=IN_CHANNELS, kernel_size=KERNEL_SIZE, out_channels=3)
python_result = own_conv.calc(tensor=input_image, kernel=kernel).squeeze().detach().numpy()


python_vec_result = own_conv.vector_calc(tensor=input_image, kernel=kernel)
# result = convolution_emulator.convolve(input_image, kernel)


# Compare with PyTorch Conv2d
conv = torch.nn.Conv2d(IN_CHANNELS, 3, kernel_size=KERNEL_SIZE, stride=1, padding=0, bias=False)
conv.weight.data = torch.tensor(kernel).float()
input_tensor = torch.tensor(input_image).reshape(BATCH_SIZE, IN_CHANNELS, 28, 28)
pytorch_result = conv(input_tensor).squeeze().detach().numpy()

# Verify results
# print("Emulator output shape:", result.sh/ape)
print("Python Emulator output shape:", python_result.shape)
print("Python Vector Emulator output shape:", python_vec_result.shape)
print("PyTorch output shape:", pytorch_result.shape)
print("Max difference pytoch vs custom python:", np.max(np.abs(python_result - pytorch_result)))
# print("Max difference pytoch vs custom C++:", np.max(np.abs(result - pytorch_result)))
# print("Max difference python vs custom C++:", np.max(np.abs(result - python_result)))
assert np.allclose(python_result, pytorch_result, atol=1e-5), "Results do not match!"
