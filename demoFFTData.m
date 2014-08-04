% demo
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';
cuda_compile('cudaFFTData',MATLAB_ROOT, CUDA_ROOT)

g = gpuDevice(1);
reset(g);
cos(gpuArray(1));

clear;
a = single(rand(16,3));
a(:,:,2) = single(rand(16,3));

bmatlab = fft2(a(:,:,1),32,16);
bmatlab(:,:,2) = fft2(a(:,:,2),32,16);
b = cudaFFTData(a,8,8); 
bg = gather(b);
figure(1); subplot(121); imagesc(real(bmatlab(:,:,1))); subplot(122); imagesc(real(bg(:,:,1)));
figure(2); imagesc(abs(bmatlab(1:17,:,1)- bg(:,:,1))); colorbar;
% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))