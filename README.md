MatlabCUDAConv
==============

Using a standard multi-threaded CPU convolution for very large kernels can be very time-consuing. This package provides a convolution using Fast Fourier Transformation implementation using CUDA.

Standard convolution can take O(nm) time compare to O((n+m) log (n+m)) where n is the data length and k is the kernel length.

## cudaConvolutionFFT.cu

Takes data, max kernel height, width, convolution kernels (multiple cells (can have different sizes)) and returns convolution results corresponding to the convolution kernels.


## Usage

1. Download the repo

```
git clone git@github.com:chrischoy/MatlabCUDAConv
```

2. Go to the repo. Open MATLAB and type

```
compile
```

3. Run demo


```
demoCudaConvolutionFFT
```

## Output

![](https://dl.dropboxusercontent.com/u/57360783/cudafft_matlabfft_conv.png)
