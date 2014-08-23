% demo
clear;
g = gpuDevice(1);
reset(g);

% matlab gpu dynamic library will be loaded.
cos(gpuArray(1));

MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';

% ld = getenv('LD_LIBRARY_PATH');
% setenv('LD_LIBRARY_PATH',[ld ':/usr/local/cuda-5.5/']);

if ismac
  MATLAB_ROOT = '/Applications/MATLAB_R2014a.app/';
  CUDA_ROOT = '/usr/local/cuda/';
end

cuda_compile('cudaConvFFTData',MATLAB_ROOT, CUDA_ROOT, 1)

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
[cvcell] = cudaConvFFTData(cuFFTedData, kernelCell, [8, 8, 8, 16]);
cvg = cvcell{1};
% cvg = gather(cv);
% dg = gather(d);
% figure(1); subplot(121); imagesc(real(matFFTedKernel(:,:,1))); subplot(122); imagesc(real(cuFFTKernel{1}(:,:,1)));

matConv = conv2(data(:,:,1),kernel(:,:,1));
for i = 2:k
  matConv(:,:,i) = conv2(data(:,:,i),kernel(:,:,i));
end

cvmatlab = sum(matConv,3);

ematlab = matFFTedKernel .* (matFFTedData);
matFFTConv = ifft2(ematlab(:,:,1));
matFFTConv(:,:,2) = ifft2(ematlab(:,:,2));


% dgc = [dg; conj([dg(end-1:-1:2, 1,:) dg(end-1:-1:2, end:-1:2,:)])];
% figure(1); subplot(121); imagesc(real(dmatlab(:,:,1))); subplot(122); imagesc(real(dgc(:,:,1)));

figure(3); subplot(131); imagesc(matConv(:,:,1)); subplot(132); imagesc(real(matFFTConv(:,:,1)));
figure(4); subplot(121); subplot(122); imagesc(real(ematlab(:,:,1)))
figure(5); subplot(121); imagesc(real(ematlab(:,:,1)));
figure(6); subplot(131); imagesc(cvg); colorbar; subplot(132); imagesc(cvg(1:n + cn - 1,1:m + cm - 1)); colorbar; subplot(133); imagesc(cvmatlab); colorbar;
figure(7); imagesc(cvg(1:n + cn - 1,1:m + cm - 1) - cvmatlab); colorbar;

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))