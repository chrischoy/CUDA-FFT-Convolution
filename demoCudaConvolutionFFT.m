% MatlabCUDAConv 
%
% To speed up convolutions, I made 

% ------------------------------------------------------------------------------
%                                                                       Compile
% ------------------------------------------------------------------------------

% Change the following lines
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';

if ismac
  MATLAB_ROOT = '/Applications/MATLAB_R2014a.app/';
  CUDA_ROOT = '/usr/local/cuda/';
end

% Debugging compile
compile
addpath('./bin')

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
m = 8; % data width
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


% ------------------------------------------------------------------------------
%                                    Matlab convolution (Conv2 and FFT versions)
% ------------------------------------------------------------------------------

% Compute convolution using FFT
% The size of ffted data should be larger than (n + cn - 1)x(m + cm - 1)
fft_h = 80;
fft_w = 16;
matFFTedData = zeros(fft_h,fft_w,k);
for i = 1:k
  matFFTedData(:,:,i) = fft2(data(:,:,i),fft_h,fft_w);
end

matFFTedKernel = zeros(fft_h, fft_w, k);
for i = 1:k
  matFFTedKernel(:,:,i) = fft2(kernel(:,:,i),fft_h,fft_w);
end

% Compute using the naive convolution
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


% ------------------------------------------------------------------------------
%                                       Convolution using GPU cudaConvolutionFFT
% ------------------------------------------------------------------------------

% You can feed multiple kernels in a cell format
kernel2 = kernel;
kernel2(1) = 100;

kernelCell = {kernel, kernel2, kernel};

thread_per_block_width = 8;
thread_per_block_height = 8;
thread_per_block_depth = 8;
thread_per_block_2d_width = 16;
threads_per_block_in =[thread_per_block_width, ...
                    thread_per_block_height, ...
                    thread_per_block_depth, ...
                    thread_per_block_2d_width];

[cvcell] = cudaConvolutionFFT(data, ... % Data 
                            cn,...      % Maximum kernel height
                            cm,...      % Maximum kernel width
                            kernelCell,...  % Multiple kernels in a cell
                            threads_per_block_in,... % threads per block
                            device_id-1); % 0-based indexing for GPU Device ID
cvg = cvcell{1}; % Get the result for the first kernel
cvg2 = cvcell{2}; % Get the result for the second kernel (kernel2)

% ------------------------------------------------------------------------------
%                                                   Comparison and visualization
% ------------------------------------------------------------------------------

% Visualize convolution result
figure(1);  subplot(131); imagesc(sum(matConv,3)); 
            subplot(132); imagesc(real(sum(matFFTConv,3)));  
            subplot(133); imagesc(real(cvg));

% Transformed data
figure(2);  imagesc(real(ematlab(:,:,1)));

% Compare matlab convolution with cuda FFT convolution
figure(3);  subplot(131); imagesc(cvg); % Convolution output ( using FFT, 
                                        % data is padded with the size of the 
                                        % kernel -1 )
            subplot(132); imagesc(cvg(1:n + cn - 1,1:m + cm - 1)); % Extract 
                                        % exact convolution part that is the 
                                        % same as matlab convolution
            subplot(133); imagesc(cvmatlab); % Visualize matlab convolution output

% Compute residual
figure(4);  imagesc(cvg(1:n + cn - 1,1:m + cm - 1) - cvmatlab); colorbar;
