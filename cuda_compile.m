function cuda_compile( func_name, matlabroot, cudaroot, optimize)
if nargin == 3
  optimize = true;
end  

% if ~verLessThan('matlab', '8.0.1')
%   % http://www.mathworks.com/help/distcomp/run-mex-functions-containing-cuda-code.html
%   setenv('MW_NVCC_PATH',[cudaroot '/nvcc'])
%   eval(sprintf('mex -v -largeArrayDims %s.cu',func_name));
% elseif isunix && ~ismac && verLessThan('matlab', '8.0.1')
  eval(['!rm ' func_name '.o']);

  if optimize
    eval(sprintf('!nvcc -O3 -DNDEBUG -Xcompiler -fPIC -v -I%s/extern/include -I%s/toolbox/distcomp/gpu/extern/include -c %s.cu', matlabroot, matlabroot, func_name));
  else
    eval(sprintf('!nvcc -O3 -DNDEBUG -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -fPIC -v -I%s/extern/include -I%s/toolbox/distcomp/gpu/extern/include -c %s.cu', matlabroot, matlabroot, func_name));
  end
  if ismac
    eval(['mex -largeArrayDims ' func_name '.o -I' cudaroot '/include -L' matlabroot '/bin/maci64  -lcudart -lcufft -lmwgpu']);  
  else
    eval(['mex -largeArrayDims ' func_name '.o -I' cudaroot '/include -L' cudaroot '/lib64 -lcudart -lcufft -L' matlabroot '/bin/glnxa64 -lmwgpu']);  
  end
end
  
% % Run system command
% !nvcc -O3 -DNDEBUG -c cudaconv.cu -Xcompiler -fPIC -I/afs/cs/package/matlab-r2013b/matlab/r2013b/extern/include -I/afs/cs/package/matlab-r2013b/matlab/r2013b/toolbox/distcomp/gpu/extern/include
% % Link object
% mex cudaconv.o -L/usr/local/cuda-6.0/lib64 -L/afs/cs/package/matlab-r2013b/matlab/r2013b/bin/glnxa64 -lcudart -lcufft -lmwgpu
% -gencode arch=compute_30,code=sm_30 