#include <cuda.h>
#include <cufft.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include "common/helper_cuda.h"
#include "cudaConvFFTData.h"


const int N_MAX_PARALLEL = 32;
static bool debug = true;

/*
 * Device Code
 */

////////////////////////////////////////////////////////////////////////////////
// Pad data with zeros, 
////////////////////////////////////////////////////////////////////////////////
__global__ void padData(
    float *d_PaddedData,
    const float *d_Data,
    int fftW,
    int fftH,
    int dataW,
    int dataH,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    if(x < fftW && y < fftH && z < FEATURE_DIM){
        if(x < dataW && y < dataH)
            d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 
                    d_Data[ IMUL(z, IMUL(dataH, dataW)) + IMUL(x, dataH ) + y];
        else
            d_PaddedData[IMUL(z, IMUL(fftW, fftH)) + IMUL(x, fftH) + y] = 0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Modulate Fourier image of padded data by Fourier image of padded kernel
// and normalize by FFT size
////////////////////////////////////////////////////////////////////////////////
__device__ void complexMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x - a.y * b.y), c * (a.y * b.x + a.x * b.y)};
    out = t;
}

__device__ void complexConjMulAndScale(cufftComplex &out, cufftComplex a, cufftComplex b, float c){
    const cufftComplex t = {c * (a.x * b.x + a.y * b.y), c * (a.y * b.x - a.x * b.y)};
    out = t;
}

__global__ void elementwiseProductAndNormalize(
    cufftComplex *fft_Output,
    const cufftComplex *fft_PaddedData,
    const cufftComplex *fft_PaddedKernel,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM,
    float scale
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;
    
    if(x < FFT_W && y < FFT_H && z < FEATURE_DIM){
        // int i = IMUL(z, IMUL(FFT_W, FFT_H)) + IMUL(FFT_H, x) + y;
        int i = z * FFT_W * FFT_H + FFT_H * x + y;
        // complexConjMulAndScale(fft_Output[i], fft_PaddedData[i], fft_PaddedKernel[i], scale);
        fft_Output[i].x = scale * (fft_PaddedData[i].x * fft_PaddedKernel[i].x - fft_PaddedData[i].y * fft_PaddedKernel[i].y);
        fft_Output[i].y = scale * (fft_PaddedData[i].y * fft_PaddedKernel[i].x + fft_PaddedData[i].x * fft_PaddedKernel[i].y);
    }
}

/* Support in-place computation, i.e. input and output can be the same */
__global__ void sumAlongFeatures(
    float *convolutionResult,
    const float *convolutionPerFeature,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;

    if(x < FFT_W && y < FFT_H){
        const int result_i = IMUL(FFT_H, x) + y;
        const int N = IMUL(FFT_W, FFT_H);

        convolutionResult[result_i] = convolutionPerFeature[result_i];
        for (int z = 1; z < FEATURE_DIM; z++){
            convolutionResult[result_i] += 
                convolutionPerFeature[IMUL(z, N) + result_i];
        }
    }
}

/*
 * Host code
 */

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ?  (a - a % b + b) : a;
}


////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    ConvPlan plan[N_MAX_PARALLEL];

    /* Declare all variables.*/
    const mxGPUArray *mxFFTData;
    const mxGPUArray *mxKernel;
    mxGPUArray *mxFFTKernel;
    mxGPUArray *mxConvolution;

    cufftComplex **d_CFFT_DATA_PER_GPU;

    /* concurrent kernel executions */
    int N_GPU; 
    int N_BATCH_PER_GPU = 2;

    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";

    /* Choose a reasonably sized number of threads for the block. */
    int THREAD_PER_BLOCK_H = 16;
    int THREAD_PER_BLOCK_W = 8;
    int THREAD_PER_BLOCK_D = 8;
    int THREAD_PER_BLOCK_2D = 32;

    // const mwSize * mxKernel_Dim;
    const mwSize * mxFFT_Dim;
    // int MblocksPerGrid, NblocksPerGrid;
    int KERNEL_H, KERNEL_W, N_KERNEL,
        CFFT_H, CFFT_W, FFT_H, FFT_W, FEATURE_DIM,
        KERNEL_SIZE, CFFT_SIZE, FFT_SIZE, CONV_SIZE;

    int gpuIdx, streamIdx, planIdx;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();
    
    /* Throw an error if the input is not a GPU array. */
    if ( (nrhs < 2) || (nrhs > 3) || !mxIsGPUArray(prhs[0]) )
        mexErrMsgIdAndTxt(errId, "The data must be FFT-ed real array in GPU");

    if (( nrhs == 3)  && mxGetNumberOfElements(prhs[2]) != 4)
        mexErrMsgIdAndTxt(errId, "CUDA Thread Size must be 4 integers : THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D\nYou must choose size such that total thread will not be larger than MaxThreadsPerBlock");

    if ( nrhs == 3 ){
        const double* threadSize = (double *)mxGetData(prhs[2]);
        THREAD_PER_BLOCK_H = (int)threadSize[0];
        THREAD_PER_BLOCK_W = (int)threadSize[1];
        THREAD_PER_BLOCK_D = (int)threadSize[2];
        THREAD_PER_BLOCK_2D = (int)threadSize[3];
        if(debug) printf("Thread size: H=%d, W=%d, D=%d, D=%d\n", THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D);
    }

    cudaDeviceProp dev;
    cudaGetDeviceProperties(&dev,0);
    int success = checkDeviceProp(dev);

    mxFFTData = mxGPUCreateFromMxArray(prhs[0]);
    mxFFT_Dim = mxGPUGetDimensions(mxFFTData);

    // FFT Dim
    // In CUDA, R2C fft will create only N/2 + 1 points. This is due to the Hermitian symmetry of the points.
    CFFT_H = mxFFT_Dim[0];
    CFFT_W = mxFFT_Dim[1];

    FFT_H = (mxFFT_Dim[0] - 1) * 2;
    FFT_W = mxFFT_Dim[1];

    FEATURE_DIM = mxFFT_Dim[2];

    CFFT_SIZE = CFFT_W * CFFT_H * FEATURE_DIM * sizeof(float2);
    FFT_SIZE  = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float);
    CONV_SIZE = FFT_W  * FFT_H  * sizeof(float);
    
    if(debug) printf("FFT Data size: h=%d, w=%d, f=%d\n", FFT_H, FFT_W, FEATURE_DIM);

    if (mxGetClassID(prhs[1]) != mxCELL_CLASS)
        mexErrMsgIdAndTxt(errId, "Kernel must be a cell array");

    mwSize nKernel = mxGetNumberOfElements(prhs[1]);
    N_KERNEL = (int)nKernel;
    plhs[0] = mxCreateCellMatrix(1, N_KERNEL);
    
    if(debug) printf("N Kernel: %d\n", N_KERNEL);


    /* Set block size and thread size */
    dim3 threadBlock3D(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    dim3 dataBlockGrid3D( iDivUp(FFT_W, threadBlock3D.x), 
                        iDivUp(FFT_H, threadBlock3D.y), 
                        iDivUp(FEATURE_DIM, threadBlock3D.z));

    dim3 threadBlock2D( THREAD_PER_BLOCK_2D, THREAD_PER_BLOCK_2D);
    dim3 dataBlockGrid2D( iDivUp(FFT_W, threadBlock2D.x), 
                        iDivUp(FFT_H, threadBlock2D.y));


    /* Find number of cuda capable devices */
    CUDA_SAFE_CALL(cudaGetDeviceCount(&N_GPU));
    if(debug) printf( "CUDA-capable device count: %i\n", N_GPU);
    
    CUDA_SAFE_CALL(cudaSetDevice(0));
    d_CFFT_DATA_PER_GPU = (cufftComplex **)malloc(N_GPU * sizeof(float));

    /*  Pad Kernel */
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_PaddedKernel,    FFT_SIZE));
    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_IFFTEProd,       FFT_SIZE));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    mwSize *FFT_dims = (mwSize *)mxMalloc(2 * sizeof(mwSize));
    FFT_dims[0] = FFT_H;
    FFT_dims[1] = FFT_W;
    FFT_dims[2] = FEATURE_DIM;

    d_CFFT_DATA_PER_GPU[0] = (cufftComplex *)mxGPUGetDataReadOnly(mxFFTData);

    // mxConvolution = mxGPUCreateGPUArray(2,
    //                         FFT_dims, // Third element will not be accessed
    //                         mxSINGLE_CLASS,
    //                         mxREAL,
    //                         MX_GPU_DO_NOT_INITIALIZE);

    // d_CONVOLUTION = (cufftReal *)(mxGPUGetData(mxConvolution));

    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_CONVOLUTION, CONV_SIZE));

    // mxFFTKernel = mxGPUCreateGPUArray(3,
    //                         mxFFT_Dim,
    //                         mxSINGLE_CLASS,
    //                         mxCOMPLEX,
    //                         MX_GPU_DO_NOT_INITIALIZE);

    // d_CFFT_KERNEL = (cufftComplex *)(mxGPUGetData(mxFFTKernel));

    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_CFFT_KERNEL, CFFT_SIZE));

    // CUDA_SAFE_CALL(cudaMalloc((void **)&d_FFTEProd, CFFT_SIZE));

    /* FFT Kernel */
    int BATCH = FEATURE_DIM;
    int FFT_Dims[] = { FFT_W, FFT_H };
    int CFFT_Dims[] = { CFFT_W, CFFT_H };

    int idist = FFT_W * FFT_H;
    int odist = CFFT_W * CFFT_H;
    
    // mwSize *FFT_dims = (mwSize *)mxMalloc(2 * sizeof(mwSize));
    //     FFT_dims[0] = FFT_H;
    //     FFT_dims[1] = FFT_W;

    N_GPU = 1;
    //Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
    for (gpuIdx = 0; gpuIdx < N_GPU; gpuIdx++)
    {
        // Set GPU
        CUDA_SAFE_CALL(cudaSetDevice(gpuIdx));
        // if (gpuIdx != 0) CUDA_SAFE_CALL();
        /* COPY mxFFTData to individual GPU */
        if (gpuIdx > 0) {
            if(debug) printf("start inter gpu copy from 0 to %d\n", gpuIdx);
            CUDA_SAFE_CALL(cudaMalloc((void **)&d_CFFT_DATA_PER_GPU[gpuIdx], CFFT_SIZE));
            CUDA_SAFE_CALL(cudaMemcpyPeerAsync(d_CFFT_DATA_PER_GPU[gpuIdx],
                    gpuIdx,
                    d_CFFT_DATA_PER_GPU[0],
                    0,
                    CFFT_SIZE,
                    plan[0].stream));
            if(debug) printf("end gpu copy from 0 to %d\n", gpuIdx);
        }

        // Set Streams
        for (streamIdx = 0; streamIdx < N_BATCH_PER_GPU; streamIdx++){
            planIdx = gpuIdx * N_BATCH_PER_GPU + streamIdx;

            CUDA_SAFE_CALL(cudaStreamCreate(&plan[planIdx].stream));
            
            // Cufft Plans
            CUFFT_SAFE_CALL(cufftPlanMany(&plan[planIdx].FFTplan_R2C, 
                2, // rank
                FFT_Dims, 
                FFT_Dims, 1, idist, // *inembed, istride, idist
                CFFT_Dims, 1, odist, // *onembed, ostride, odist
                CUFFT_R2C, 
                BATCH)); // batch
            cufftSetStream(plan[planIdx].FFTplan_R2C, plan[planIdx].stream);

            CUFFT_SAFE_CALL(cufftPlanMany(&plan[planIdx].FFTplan_C2R, 
                2, // rank
                FFT_Dims,
                CFFT_Dims, 1, odist, // *inembed, istride, idist
                FFT_Dims, 1, idist, // *onembed, ostride, odist
                CUFFT_C2R, 
                BATCH)); // batch
            cufftSetStream(plan[planIdx].FFTplan_C2R, plan[planIdx].stream);

            plan[planIdx].d_CFFT_DATA = d_CFFT_DATA_PER_GPU[gpuIdx];

            //Allocate memory
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_CFFT_KERNEL, CFFT_SIZE));
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_FFTEProd,    CFFT_SIZE));
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_CONVOLUTION, CONV_SIZE));
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_IFFTEProd,       FFT_SIZE));
            // d_Kernel, dynamically set
            CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_PaddedKernel,    FFT_SIZE));
            // h_Kernel, dynamically set
            // CUDA_SAFE_CALL(cudaMallocHost((void **)&plan[planIdx].h_CONVOLUTION,    CONV_SIZE));
        }
    }
    

    /* For each kernel iterate */
    int N_PLANS = N_GPU * N_BATCH_PER_GPU;
    printf("N Plans %d\n",N_PLANS);

    int kernelIdx = 0;
    int lastPlanIdx;

    while(kernelIdx < N_KERNEL){
        if(debug) printf( "Kernel: %d\n",kernelIdx);

        for (gpuIdx = 0; gpuIdx < N_GPU; gpuIdx++){
            if (kernelIdx >= N_KERNEL) break;

            // Set GPU
            CUDA_SAFE_CALL(cudaSetDevice(gpuIdx));
            
            // Set Streams
            for (streamIdx = 0; streamIdx < N_BATCH_PER_GPU; streamIdx++){
                planIdx = gpuIdx * N_BATCH_PER_GPU + streamIdx;

                // Get Kernel Data
                const mxArray *mxCurrentCell = mxGetCell(prhs[1], kernelIdx);
                {
                    if( mxGetClassID(mxCurrentCell) != mxSINGLE_CLASS || mxGetNumberOfDimensions(mxCurrentCell) != 3 )
                        mexErrMsgIdAndTxt(errId, "Kernels must be of type float and have features larger than 1");

                    if(debug) printf("Start plan %d\n", planIdx);

                    plan[planIdx].h_Kernel = (float *)mxGetData(mxCurrentCell);
                    plan[planIdx].mxKernel_Dim = mxGetDimensions(mxCurrentCell);

                    // Kernel dimensions
                    KERNEL_H = plan[planIdx].mxKernel_Dim[0];
                    KERNEL_W = plan[planIdx].mxKernel_Dim[1];
                    KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);

                    if(debug) printf("Start copy\n");
                    // CUDA_SAFE_CALL(cudaHostRegister(plan[planIdx].h_Kernel, KERNEL_SIZE, cudaHostRegisterPortable));
                    // CUDA_SAFE_CALL(cudaHostGetDevicePointer((void **) &plan[planIdx].d_Kernel, (void *)plan[planIdx].h_Kernel, 0));
                    CUDA_SAFE_CALL(cudaMalloc((void **)&plan[planIdx].d_Kernel, KERNEL_SIZE));
                    CUDA_SAFE_CALL(cudaMemcpyAsync(plan[planIdx].d_Kernel, plan[planIdx].h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice, plan[planIdx].stream));
                    // CUDA_SAFE_CALL(cudaMemcpy(plan[planIdx].d_Kernel, plan[planIdx].h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
                    mxKernel = NULL;
                }

                if(debug) printf("Kernel size: h=%d, w=%d\n", KERNEL_H, KERNEL_W);

                if (FEATURE_DIM != plan[planIdx].mxKernel_Dim[2] || KERNEL_W > FFT_W || KERNEL_H > FFT_H ){
                    mexErrMsgIdAndTxt(errId, "Kernel and Data must have the same number of features and kernel size should be smaller than data size");
                }

                // CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));
                if(debug) printf("Sync before padding\n");
                padData<<<dataBlockGrid3D, threadBlock3D, 0, plan[planIdx].stream>>>(
                    plan[planIdx].d_PaddedKernel,
                    plan[planIdx].d_Kernel,
                    FFT_W,
                    FFT_H,
                    KERNEL_W,
                    KERNEL_H,
                    FEATURE_DIM
                    );
                if(debug) printf("Padding done\n");

                CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));
                CUFFT_SAFE_CALL(cufftExecR2C(plan[planIdx].FFTplan_R2C, plan[planIdx].d_PaddedKernel, plan[planIdx].d_CFFT_KERNEL));
                // CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));

                if(debug) printf("FFT done\n");
                
                /* Hadamard product, Element-wise multiplication in frequency domain */
                /* If execute the following, second compile of this file create MATLAB error */
                elementwiseProductAndNormalize<<<dataBlockGrid3D, threadBlock3D, 0, plan[planIdx].stream>>>(
                        plan[planIdx].d_FFTEProd, // out
                        plan[planIdx].d_CFFT_DATA, // in data
                        plan[planIdx].d_CFFT_KERNEL,   // in kernel
                        CFFT_H,
                        CFFT_W,
                        FEATURE_DIM,
                        1.0f / (FFT_W * FFT_H)
                    );
                if(debug) printf("Eprod done\n");
                CUFFT_SAFE_CALL(cufftExecC2R(plan[planIdx].FFTplan_C2R, plan[planIdx].d_FFTEProd, plan[planIdx].d_IFFTEProd));
                // CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));
                if(debug) printf("Second fft done\n");
                sumAlongFeatures<<<dataBlockGrid2D, threadBlock2D, 0, plan[planIdx].stream>>>(
                        plan[planIdx].d_CONVOLUTION,
                        plan[planIdx].d_IFFTEProd,
                        FFT_H,
                        FFT_W,
                        FEATURE_DIM
                    );
                if(debug) printf("sum along features done\n");
                // CUDA_SAFE_CALL(cudaHostUnregister(plan[planIdx].h_Kernel));

                plan[planIdx].convolutionResult = mxCreateNumericArray(2, FFT_dims, mxSINGLE_CLASS, mxREAL);
                plan[planIdx].h_CONVOLUTION = (float *)mxGetData(plan[planIdx].convolutionResult);

                // CUDA_SAFE_CALL(cudaHostRegister(plan[planIdx].h_CONVOLUTION, CONV_SIZE, cudaHostRegisterPortable));
                CUDA_SAFE_CALL(cudaMemcpyAsync(plan[planIdx].h_CONVOLUTION, plan[planIdx].d_CONVOLUTION, CONV_SIZE ,cudaMemcpyDeviceToHost, plan[planIdx].stream));
                // CUDA_SAFE_CALL(cudaMemcpy(plan[planIdx].h_CONVOLUTION, plan[planIdx].d_CONVOLUTION, CONV_SIZE ,cudaMemcpyDeviceToHost));

                if(debug) printf("Copy done\n");
 
                // CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));
                if(debug) printf("Sync done\n");

                mxSetCell(plhs[0], kernelIdx, plan[planIdx].convolutionResult);
                if(debug) printf("Setting Cell done\n");
                // if(debug){
                //     for(int i = 0; i < 10; i++)
                //         printf("%f\n", plan[planIdx].h_CONVOLUTION[i]);
                // }
                kernelIdx = kernelIdx + 1;
                if (kernelIdx >= N_KERNEL) break;
            }
        }

        lastPlanIdx = planIdx;
        if(debug) printf("lastPlanIdx : %d\n", lastPlanIdx);

        for (gpuIdx = 0; gpuIdx < N_GPU; gpuIdx++){
            if (planIdx > lastPlanIdx ) break;

            // Set GPU
            CUDA_SAFE_CALL(cudaSetDevice(gpuIdx));
            
            // Set Streams
            for (streamIdx = 0; streamIdx < N_BATCH_PER_GPU; streamIdx++){
                planIdx = gpuIdx * N_BATCH_PER_GPU + streamIdx;
                if (planIdx > lastPlanIdx ) break;
                CUDA_SAFE_CALL(cudaStreamSynchronize(plan[planIdx].stream));
                CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_Kernel));
                // CUDA_SAFE_CALL(cudaHostUnregister(plan[planIdx].h_Kernel));
                // CUDA_SAFE_CALL(cudaHostUnregister(plan[planIdx].h_CONVOLUTION));
                if(debug) printf("Synchronize %d\n", planIdx);
            }
        }
    }
    
    // plhs[1] = mxGPUCreateMxArrayOnGPU(mxFFTKernel);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(mxFFTData);
    // mxGPUDestroyGPUArray(mxConvolution);    
    // mxGPUDestroyGPUArray(mxFFTKernel);

    // if(mxKernel == NULL) mxGPUDestroyGPUArray(mxKernel);

    for ( gpuIdx = 0; gpuIdx < N_GPU; gpuIdx++)
    {
        // Set GPU
        CUDA_SAFE_CALL(cudaSetDevice(gpuIdx));
        if(debug) printf( "free DATA per GPU %d\n", gpuIdx);
        CUDA_SAFE_CALL(cudaFree(d_CFFT_DATA_PER_GPU[gpuIdx]));
        // Set Streams
        for (int streamIdx = 0; streamIdx < N_BATCH_PER_GPU; streamIdx++){
            int planIdx = gpuIdx * N_BATCH_PER_GPU + streamIdx;
            
            cufftDestroy(plan[planIdx].FFTplan_R2C);
            cufftDestroy(plan[planIdx].FFTplan_C2R);

            if(debug) printf( "free plans\n");

            //Allocate memory
            CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_CFFT_KERNEL));
            CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_FFTEProd));
            CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_CONVOLUTION));
            CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_IFFTEProd));
            // d_Kernel
            CUDA_SAFE_CALL(cudaFree(plan[planIdx].d_PaddedKernel));
            // h_Kernel
            // CUDA_SAFE_CALL(cudaFreeHost(plan[planIdx].h_CONVOLUTION));
            if(debug) printf( "free stream\n");
            CUDA_SAFE_CALL(cudaStreamDestroy(plan[planIdx].stream));
        }

        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
    }
    
    // // if(mxKernel == NULL) cudaFree(d_Kernel);

    mxFree(FFT_dims);
}
