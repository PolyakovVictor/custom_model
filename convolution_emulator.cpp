#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE 28
#define KERNEL_SIZE 3
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1) // 26 for 28x28 input and 3x3 kernel


// Simple 2D convolution emulator
void convolve(float* input, float* kernel, float* output) {
    if (!input || !kernel || !output) {
        printf("Error: one of the inputs is null\n");
        return;
    }

    // Печать входного изображения (только центральные 5x5)
    printf("\n--- INPUT (центр 5x5) ---\n");
    for (int i = 11; i < 16; i++) {
        for (int j = 11; j < 16; j++) {
            printf("%6.2f ", input[i * INPUT_SIZE + j]);
        }
        printf("\n");
    }

    // Печать ядра
    printf("\n--- KERNEL ---\n");
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            printf("%6.3f ", kernel[i * KERNEL_SIZE + j]);
        }
        printf("\n");
    }

    // Выполнение свёртки
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float sum = 0.0f;
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    int input_i = i + ki;
                    int input_j = j + kj;
                    if (input_i < INPUT_SIZE && input_j < INPUT_SIZE) {
                        sum += input[input_i * INPUT_SIZE + input_j] * kernel[ki * KERNEL_SIZE + kj];
                    }
                }
            }
            output[i * OUTPUT_SIZE + j] = sum;
        }
    }

    // Печать выходного изображения (только центральные 5x5)
    printf("\n--- OUTPUT (центр 5x5) ---\n");
    for (int i = 10; i < 15; i++) {
        for (int j = 10; j < 15; j++) {
            printf("%6.2f ", output[i * OUTPUT_SIZE + j]);
        }
        printf("\n");
    }

    printf("\n[DEBUG] Convolution completed\n");
}


// PyBind11 wrapper
namespace py = pybind11;

py::array_t<float> convolve_wrapper(py::array_t<float> input, py::array_t<float> kernel) {
    // Check input dimensions
    py::buffer_info input_info = input.request();
    py::buffer_info kernel_info = kernel.request();

    if (input_info.shape[0] != INPUT_SIZE || input_info.shape[1] != INPUT_SIZE) {
        throw std::runtime_error("Input must be 28x28");
    }
    if (kernel_info.shape[0] != KERNEL_SIZE || kernel_info.shape[1] != KERNEL_SIZE) {
        throw std::runtime_error("Kernel must be 3x3");
    }

    // Allocate output array
    py::array_t<float> output({OUTPUT_SIZE, OUTPUT_SIZE});
    py::buffer_info output_info = output.request();

    // Perform convolution
    convolve(static_cast<float*>(input_info.ptr), static_cast<float*>(kernel_info.ptr),
             static_cast<float*>(output_info.ptr));

    return output;
}

PYBIND11_MODULE(convolution_emulator, m) {
    m.def("convolve", &convolve_wrapper, "Emulate 2D convolution on input image with kernel");
}