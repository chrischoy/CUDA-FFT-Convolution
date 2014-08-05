% demo
clear;
g = gpuDevice(1);
reset(g);

% matlab gpu dynamic library will be loaded.
cos(gpuArray(1));

MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';
cuda_compile('cudaConvFFTData',MATLAB_ROOT, CUDA_ROOT, 1)

clear;
n = 64;
m = 8;
k = 3;

cn = 3;
cm = 3;
a = single(rand(n,m));
for i = 2:k
  a(:,:,i) = single(rand(n,m));
end

c = zeros(3,3,k,'single');
c(:,:,1) = single([1 2 3;4 5 6; 7 8 9]);
for i = 2:k
  c(:,:,i) = single(rand(cn,cm));
end

a(5:7,2:4,1) = c(:,:,1);
a(21:23,1:3,2) = c(:,:,1);
a(1:3,m-2:m,k) = c(:,:,1);
c(:,:,k) = c(:,:,1);

bmatlab = fft2(a(:,:,1),2*n,2*m);
bmatlab(:,:,2) = fft2(a(:,:,2),2*n,2*m);

b = cudaFFTData(a, cn,cm);
% b = gpuArray(bmatlab(1:25,:,:));


dmatlab = fft2(c(:,:,1),2*n,16);
dmatlab(:,:,2) = fft2(c(:,:,2),2*n,16);

% Hadamard product
ematlab = dmatlab .* (bmatlab);
gc = gpuArray(single(c));
gccell = {gc, gc+1};
% [cv, d] = cudaConvFFTData(b,gc);
cv = cudaConvFFTData(b,gc);
cvg = gather(cv);
% dg = gather(d);


e = conv2(a(:,:,1),c(:,:,1));
for i = 2:k
  e(:,:,i) = conv2(a(:,:,i),c(:,:,i));
end

cvmatlab = sum(e,3);

eifftmatlab = ifft2(ematlab(:,:,1));
eifftmatlab(:,:,2) = ifft2(ematlab(:,:,2));


% dgc = [dg; conj([dg(end-1:-1:2, 1,:) dg(end-1:-1:2, end:-1:2,:)])];
% figure(1); subplot(121); imagesc(real(dmatlab(:,:,1))); subplot(122); imagesc(real(dgc(:,:,1)));

figure(3); subplot(131); imagesc(e(:,:,1)); subplot(132); imagesc(real(eifftmatlab(:,:,1)));
figure(4); subplot(121); subplot(122); imagesc(real(ematlab(:,:,1)))
figure(5); subplot(121); imagesc(real(ematlab(:,:,1)));
figure(6); subplot(131); imagesc(cvg); colorbar; subplot(132); imagesc(cvg(1:n + cn - 1,1:m + cm - 1)); colorbar; subplot(133); imagesc(cvmatlab); colorbar;
figure(7); imagesc(cvg(1:n + cn - 1,1:m + cm - 1) - cvmatlab); colorbar;

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))