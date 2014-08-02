% demo
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';
cuda_compile('cudaConv',MATLAB_ROOT, CUDA_ROOT)

gpuArray(1);


a = rand(5);
a(:,:,2) = rand(5);

bmatlab = fft2(a(:,:,1),16,16);
b = cudaFFTData(single(a),8,8); subplot(121); imagesc(imag(bmatlab)); subplot(122); imagesc(imag(b(:,:,1)));

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))