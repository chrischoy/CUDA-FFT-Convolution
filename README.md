MatlabCUDAConv
==============

MATLAB wrapped convolution function using CUDA

cudaFFTData.cu

IN : host float
        k-dim m x n data 
     host int
        max m kernel
     host int
        max n kernel

OUT : device complex float
        k-dim close multiple of 16 ( m + max p ) close multiple of 16 ( n + max q )

cudaConv.cu
 
IN : device float
        k-dim close multiple of 16 ( m + max p ) close multiple of 16 ( n + max q )

     host float
        k-dim p x q kernel

OUT : device float
        close multiple of 16 ( m + max p ) close multiple of 16 ( n + max q )

      [device float
        kernel in GPU.]

