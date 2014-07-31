function cuda_compile( func_name, matlabroot, cudaroot)
  eval(['!rm ' func_name '.o']);
  eval(sprintf('!nvcc -Xcompiler -fPIC -I%s/extern/include -I%s/toolbox/distcomp/gpu/extern/include -c %s.cu', matlabroot, matlabroot, func_name));
  mex([func_name '.o'], ['-I' cudaroot '/include'], ['-L' cudaroot '/lib64'],['-L' matlabroot '/bin/glnxa64 -lmwgpu'], '-lcudart -lcufft');
end

% % Run system command
% !nvcc -O3 -DNDEBUG -c cudaconv.cu -Xcompiler -fPIC -I/afs/cs/package/matlab-r2013b/matlab/r2013b/extern/include -I/afs/cs/package/matlab-r2013b/matlab/r2013b/toolbox/distcomp/gpu/extern/include
% % Link object
% mex cudaconv.o -L/usr/local/cuda-6.0/lib64 -L/afs/cs/package/matlab-r2013b/matlab/r2013b/bin/glnxa64 -lcudart -lcufft -lmwgpu
