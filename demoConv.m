% demo
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';
cuda_compile('cudaConv',MATLAB_ROOT, CUDA_ROOT)
clear;

c = gpuArray(1);
cos(c);
clear;
n = 64;
m = 8;
cn = 3;
cm = 3;
a = single(rand(n,m));
a(:,:,2) = single(rand(n,m));

c = [1 2 3;4 5 6; 7 8 9];
c(:,:,2) = c;

a(5:7,2:4,1) = c(:,:,1);
a(21:23,1:3,2) = c(:,:,1);

bmatlab = fft2(a(:,:,1),2*n,2*m);
bmatlab(:,:,2) = fft2(a(:,:,2),2*n,2*m);

b = cudaFFTData(a, 3,3);
% b = gpuArray(bmatlab(1:25,:,:));


dmatlab = fft2(c(:,:,1),2*n,16);
dmatlab(:,:,2) = fft2(c(:,:,2),2*n,16);

% Hadamard product
ematlab = dmatlab .* (bmatlab);
[cv, d] = cudaConv(b,single(c));
dg = gather(d);
e = conv2(a(:,:,1),c(:,:,1));
e(:,:,2) = conv2(a(:,:,2),c(:,:,2));
cvmatlab = sum(e,3);
eifftmatlab = ifft2(ematlab(:,:,1));
eifftmatlab(:,:,2) = ifft2(ematlab(:,:,2));


dgc = [dg; conj([dg(end-1:-1:2, 1,:) dg(end-1:-1:2, end:-1:2,:)])];

figure(1); subplot(121); imagesc(real(dmatlab(:,:,1))); subplot(122); imagesc(real(dgc(:,:,1)));
figure(3); subplot(131); imagesc(e(:,:,1)); subplot(132); imagesc(real(eifftmatlab(:,:,1)));
figure(4); subplot(121); subplot(122); imagesc(real(ematlab(:,:,1)))
figure(5); subplot(121); imagesc(real(ematlab(:,:,1))); 
figure(6); subplot(131); imagesc(cv); colorbar; subplot(132); imagesc(cv(1:n + cn - 1,1:m + cm - 1)); colorbar; subplot(133); imagesc(cvmatlab); colorbar;

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))