MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.5/';

if ismac
  MATLAB_ROOT = '/Applications/MATLAB_R2014a.app/';
  CUDA_ROOT = '/usr/local/cuda/';
end
cuda_compile('src/cudaFFTData.cu',MATLAB_ROOT, CUDA_ROOT, './bin/', true)
cuda_compile('src/cudaConvFFTData.cu',MATLAB_ROOT, CUDA_ROOT, './bin/',true)
cuda_compile('src/cudaConvolutionFFT.cu',MATLAB_ROOT, CUDA_ROOT, './bin/',true)
