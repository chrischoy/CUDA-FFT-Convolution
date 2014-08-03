% demo
MATLAB_ROOT = '/afs/cs/package/matlab-r2013b/matlab/r2013b/';
CUDA_ROOT = '/usr/local/cuda-6.0/';
cuda_compile('cudaConv',MATLAB_ROOT, CUDA_ROOT)
clear;

c = gpuArray(1);
cos(c);
clear;
a = single(rand(16,8));
a(:,:,2) = single(rand(16,8));

c = [1 2 3;4 5 6; 7 8 9];
c(:,:,2) = c;

a(5:7,2:4,:) = c;

bmatlab = fft2(a(:,:,1),32,16);
bmatlab(:,:,2) = fft2(a(:,:,2),32,16);
b = gpuArray(bmatlab(1:17,:,:));



dmatlab = fft2(c(:,:,1),32,16);
dmatlab(:,:,2) = fft2(c(:,:,2),32,16);

ematlab = dmatlab .* conj(bmatlab);
[cv, d, eprod, iffteprod] = cudaConv(b,single(c));
eprodg = gather(eprod);
dg = gather(d);
iffteprodg = gather(iffteprod);
e = conv2(a(:,:,1),c(:,:,1));
e(:,:,2) = conv2(a(:,:,2),c(:,:,2));
cvmatlab = sum(e,3);
eifftmatlab = ifft2(ematlab(:,:,1));
eifftmatlab(:,:,2) = ifft2(ematlab(:,:,2));


dgc = [dg; conj([dg(end-1:-1:2, 1,:) dg(end-1:-1:2, end:-1:2,:)])];
eprodgc = [eprodg; conj([eprodg(end-1:-1:2, 1,:) eprodg(end-1:-1:2, end:-1:2,:)])] * 32 * 16;

figure(1); subplot(121); imagesc(real(dmatlab(:,:,1))); subplot(122); imagesc(real(dgc(:,:,1)));
% figure(2); imagesc(abs(dmatlab(:,:,1)- dgc(:,:,1))); colorbar;
figure(3); subplot(131); imagesc(e(:,:,1)); subplot(132); imagesc(real(eifftmatlab(:,:,1))); subplot(133); imagesc(iffteprodg(:,:,1));
figure(4); subplot(121); imagesc(real(eprodg(:,:,1))); subplot(122); imagesc(real(ematlab(:,:,1)))
figure(5); subplot(121); imagesc(real(ematlab(:,:,1))); subplot(122); imagesc(real(eprodg(:,:,1))); colorbar;
figure(6); subplot(121); imagesc(cv); colorbar; subplot(122); imagesc(cvmatlab); colorbar;

% k = rand(5);
% k(:,:,2) = rand(5);
% [c] = cudaConv(b,single(k))