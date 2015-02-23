% MatlabCUDAConv 
%
% To speed up convolutions, I made 

% ------------------------------------------------------------------------------
%                                                                       Compile
% ------------------------------------------------------------------------------

% Change the following lines
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.5/';

if ismac
  MATLAB_ROOT = '/Applications/MATLAB_R2014a.app/';
  CUDA_ROOT = '/usr/local/cuda/';
end

% Debugging compile
cuda_compile('src/cudaConvolutionFFT',MATLAB_ROOT, CUDA_ROOT, './bin', 0); 

% ------------------------------------------------------------------------------
%                                                                  Clear the GPU
% ------------------------------------------------------------------------------

clear;
device_id = 1; % 1-base GPU index (MATLAB convention)
g = gpuDevice(device_id);
reset(g);
cos(gpuArray(1)); % force matlab gpu dynamic library loading


% ------------------------------------------------------------------------------
%                                                              Experiment setup
% ------------------------------------------------------------------------------

n = 64;  % data height
m = 105; % data width
k = 5;   % number of channels

cn = 10; % kernel height
cm = 4;  % kernel width

% Make random data
data = single(rand(n,m));
for i = 2:k
  data(:,:,i) = single(rand(n,m));
end

% Make random kernel
kernel = zeros(cn,cm,k,'single');
kernel(:,:,1) = single(reshape(1:cn*cm,cn,cm));
for i = 2:k
  kernel(:,:,i) = single(rand(cn,cm));
end

% To verify experiment, put kernel values to specific regions
data(5:(4+cn),2:(1+cm),1) = kernel(:,:,1);
data(21:(20+cn),1:cm,2) = kernel(:,:,1);
data(1:cn,(m-(cm-1)):m,k) = kernel(:,:,1);
kernel(:,:,k) = kernel(:,:,1);

% ------------------------------------------------------------------------------
%                                                         Flip Kernel (Required)
% ------------------------------------------------------------------------------

for i = 1:k
  kernel(:,:,i) = kernel(end:-1:1,end:-1:1,i);
end


% TODO
matFFTedData = zeros(80,16,k);
for i = 1:k
  matFFTedData(:,:,i) = fft2(data(:,:,i),80,16);
end

matFFTedKernel = zeros(n + 16, 16, k);
for i = 1:k
  matFFTedKernel(:,:,i) = fft2(kernel(:,:,i),80,16);
end
kernelCell = {kernel, kernel, kernel};

[cvcell] = cudaConvolutionFFT(data, cn, cm, kernelCell, [8, 8, 8, 16], device_id-1);
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
for i=1:k
    matFFTConv(:,:,i) = ifft2(ematlab(:,:,i));
end


% dgc = [dg; conj([dg(end-1:-1:2, 1,:) dg(end-1:-1:2, end:-1:2,:)])];
% figure(1); subplot(121); imagesc(real(dmatlab(:,:,1))); subplot(122); imagesc(real(dgc(:,:,1)));

figure(1); subplot(131); imagesc(sum(matConv,3)); subplot(132); imagesc(real(sum(matFFTConv,3)));  subplot(133); imagesc(real(cvg));
figure(2); subplot(121); subplot(122); imagesc(real(ematlab(:,:,1)))
figure(3); subplot(121); imagesc(real(ematlab(:,:,1)));
figure(4); subplot(131); imagesc(cvg); colorbar; subplot(132); imagesc(cvg(1:n + cn - 1,1:m + cm - 1)); colorbar; subplot(133); imagesc(cvmatlab); colorbar;
figure(5); imagesc(cvg(1:n + cn - 1,1:m + cm - 1) - cvmatlab); colorbar;

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))
