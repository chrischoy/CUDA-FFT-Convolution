% demo
clear;
g = gpuDevice(1);
reset(g);

% matlab gpu dynamic library will be loaded.
cos(gpuArray(1));

MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-5.5/';

if ismac
  MATLAB_ROOT = '/Applications/MATLAB_R2014a.app/';
  CUDA_ROOT = '/usr/local/cuda/';
end

cuda_compile('cudaConvFFTDataStreams',MATLAB_ROOT, CUDA_ROOT, 0)

clear;
n = 64;
m = 8;
k = 5;

cn = 3;
cm = 3;
data = single(rand(n,m));
for i = 2:k
  data(:,:,i) = single(rand(n,m));
end

kernel = zeros(3,3,k,'single');
kernel(:,:,1) = single([1 2 3;4 5 6; 7 8 9]);
for i = 2:k
  kernel(:,:,i) = single(rand(cn,cm));
end

data(5:7,2:4,1) = kernel(:,:,1);
data(21:23,1:3,2) = kernel(:,:,1);
data(1:3,m-2:m,k) = kernel(:,:,1);
kernel(:,:,k) = kernel(:,:,1);


for i = 1:k
  kernel(:,:,i) = kernel(end:-1:1,end:-1:1,i);
end

matFFTedData = zeros(80,16,k);
for i = 1:k
  matFFTedData(:,:,i) = fft2(data(:,:,i),80,16);
end

cuFFTedData = cudaFFTData(data, cn,cm);
% b = gpuArray(bmatlab(1:25,:,:));


matFFTedKernel = zeros(n + 16, 16, k);
for i = 1:k
  matFFTedKernel(:,:,i) = fft2(kernel(:,:,i),80,16);
end

% Hadammard product
gpuKernel = gpuArray(single(kernel));
gpuKernelCell = {gpuKernel};
kernelCell = {kernel, kernel, kernel};
% cvcell = cudaConvFFTData(b,gccell);
% [cvcell] = cudaConvFFTDataStreams(cuFFTedData, kernelCell, [8, 8, 8, 16]);
a = cudaConvFFTDataStreams(cuFFTedData, kernelCell, [8, 8, 8, 16],[cn, cm]);
% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))