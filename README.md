CUDA-FFT-Convolution
==============

Using a standard multi-threaded CPU convolution for very large kernels is very inefficient and slow. This package provides GPU convolution using Fast Fourier Transformation implementation using CUDA.

Standard convolution in time domain takes O(nm) time whereas convolution in frequency domain takes O((n+m) log (n+m)) time where n is the data length and k is the kernel length.

## cudaConvolutionFFT.cu

The main file takes data, max kernel height, width, convolution kernels (multiple kernels in cell format) and returns convolution results that corresponds to the convolution kernels.

## Usage and Instructions

1. Download the repo.

    ```
    git clone http://github.com/chrischoy/MatlabCUDAConv
    ```

2. Go to the repo. Open MATLAB and type

    ```
    compile
    ```

3. Run demo. the demo file `demoCudaConvolutionFFT.m` contains a detailed instruction and demo usage


    ```
    demoCudaConvolutionFFT
    ```

## Output

![](https://dl.dropboxusercontent.com/u/57360783/cudafft_matlabfft_conv.png)

### More resource

[http://chrischoy.org/projects/cuda-fft-convolution](http://chrischoy.org/projects/cuda-fft-convolution)
