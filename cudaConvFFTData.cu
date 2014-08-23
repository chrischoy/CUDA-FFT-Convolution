#include <cuda.h>
#include <cufft.h>
#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include "common/helper_cuda.h"
#include "cudaConvFFTData.h"
#include "cudaConvFFTData.cuh"


const int N_MAX_PARALLEL = 32;
static bool debug = false;

enum OUT_INDEX{
    CONVOLUTION_CELL_INDEX
};

enum IN_INDEX{
    FFT_DATA_INDEX,
    KERNLE_CELL_INDEX,
    THREAD_SIZE_INDEX // Optional
};

////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    const mxGPUArray *mxFFTData;
    const mxGPUArray *mxKernel;
    mxGPUArray *mxFFTKernel;
    mxGPUArray *mxConvolution;
    mxArray *convolutionResult;
    
    /* cufftComplex is float2 */
    const cufftComplex *d_CFFT_DATA;
    cufftComplex *d_CFFT_KERNEL;
    cufftComplex *d_FFTEProd;

    float *d_CONVOLUTION;
    float *d_IFFTEProd;

    float *h_Kernel;
    float *h_CONVOLUTION;
    float *d_Kernel;
    float *d_PaddedKernel;

    char const * const errId = "cudaConvFFTData:InvalidInput";

    /* Choose a reasonably sized number of threads for the block. */
    int THREAD_PER_BLOCK_H = 16;
    int THREAD_PER_BLOCK_W = 8;
    int THREAD_PER_BLOCK_D = 8;
    int THREAD_PER_BLOCK_2D = 32;

    const mwSize * mxKernel_Dim;
    const mwSize * mxFFT_Dim;
    // int MblocksPerGrid, NblocksPerGrid;
    int KERNEL_H, KERNEL_W, N_KERNEL,
        CFFT_H, CFFT_W, FFT_H, FFT_W, FEATURE_DIM,
        KERNEL_SIZE, CFFT_SIZE, FFT_SIZE, CONV_SIZE;

    /* Initialize the MathWorks GPU API. */
    // If initialized mxInitGPU do nothing
    if (mxInitGPU() != MX_GPU_SUCCESS)
        mexErrMsgTxt("mxInitGPU fail");
    
    /* Throw an error if the input is not a GPU array. */
    if ( (nrhs < 2) || (nrhs > 3) || !mxIsGPUArray(prhs[FFT_DATA_INDEX]) )
        mexErrMsgIdAndTxt(errId, "The data must be FFT-ed real array in GPU");

    if (( nrhs == 3)  && mxGetNumberOfElements(prhs[THREAD_SIZE_INDEX]) != 4)
        mexErrMsgIdAndTxt(errId, "CUDA Thread Size must be 4 integers : THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D\nYou must choose size such that total thread will not be larger than MaxThreadsPerBlock");

    if ( nrhs == 3 ){
        const double* threadSize = (double *)mxGetData(prhs[THREAD_SIZE_INDEX]);
        THREAD_PER_BLOCK_H = (int)threadSize[0];
        THREAD_PER_BLOCK_W = (int)threadSize[1];
        THREAD_PER_BLOCK_D = (int)threadSize[2];
        THREAD_PER_BLOCK_2D = (int)threadSize[3];
        if(debug) fprintf(stderr,"Thread size: H=%d, W=%d, D=%d, 2D=%d\n", THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D, THREAD_PER_BLOCK_2D);
    }

    // cudaDeviceProp dev;
    // cudaGetDeviceProperties(&dev,0);
    // int success = checkDeviceProp(dev);

    mxFFTData = mxGPUCreateFromMxArray(prhs[FFT_DATA_INDEX]);
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
    
    if(debug) fprintf(stderr,"FFT Data size: h=%d, w=%d, f=%d\n", FFT_H, FFT_W, FEATURE_DIM);

    if (mxGetClassID(prhs[KERNLE_CELL_INDEX]) != mxCELL_CLASS)
        mexErrMsgIdAndTxt(errId, "Kernel must be a cell array");

    mwSize nKernel = mxGetNumberOfElements(prhs[KERNLE_CELL_INDEX]);
    N_KERNEL = (int)nKernel;
    plhs[CONVOLUTION_CELL_INDEX] = mxCreateCellMatrix(1, N_KERNEL);

    if(debug) fprintf(stderr,"N Kernel: %d\n", N_KERNEL);


    /* Set block size and thread size */
    dim3 threadBlock3D(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    dim3 dataBlockGrid3D( iDivUp(FFT_W, threadBlock3D.x), 
                        iDivUp(FFT_H, threadBlock3D.y), 
                        iDivUp(FEATURE_DIM, threadBlock3D.z));

    dim3 threadBlock2D( THREAD_PER_BLOCK_2D, THREAD_PER_BLOCK_2D);
    dim3 dataBlockGrid2D( iDivUp(FFT_W, threadBlock2D.x), 
                        iDivUp(FFT_H, threadBlock2D.y));


    /*  Pad Kernel */
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_PaddedKernel,    FFT_SIZE));
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_IFFTEProd,       FFT_SIZE));

    /* Create a GPUArray to hold the result and get its underlying pointer. */
    // mwSize *FFT_dims = (mwSize *)mxMalloc(2 * sizeof(mwSize));
    // FFT_dims[0] = FFT_H;
    // FFT_dims[1] = FFT_W;
    // FFT_dims[2] = FEATURE_DIM;

    d_CFFT_DATA = (cufftComplex *)mxGPUGetDataReadOnly(mxFFTData);

    // mxConvolution = mxGPUCreateGPUArray(2,
    //                         FFT_dims, // Third element will not be accessed
    //                         mxSINGLE_CLASS,
    //                         mxREAL,
    //                         MX_GPU_DO_NOT_INITIALIZE);

    // d_CONVOLUTION = (cufftReal *)(mxGPUGetData(mxConvolution));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_CONVOLUTION, CONV_SIZE));

    // mxFFTKernel = mxGPUCreateGPUArray(3,
    //                         mxFFT_Dim,
    //                         mxSINGLE_CLASS,
    //                         mxCOMPLEX,
    //                         MX_GPU_DO_NOT_INITIALIZE);

    // d_CFFT_KERNEL = (cufftComplex *)(mxGPUGetData(mxFFTKernel));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_CFFT_KERNEL, CFFT_SIZE));

    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_FFTEProd, CFFT_SIZE));

    /* FFT Kernel */
    int BATCH = FEATURE_DIM;
    int FFT_Dims[] = { FFT_W, FFT_H };
    int CFFT_Dims[] = { CFFT_W, CFFT_H };

    int idist = FFT_W * FFT_H;
    int odist = CFFT_W * CFFT_H;

    cufftHandle FFTplan_R2C, FFTplan_C2R;
    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_R2C, 
        2, // rank
        FFT_Dims, 
        FFT_Dims, 1, idist, // *inembed, istride, idist
        CFFT_Dims, 1, odist, // *onembed, ostride, odist
        CUFFT_R2C, 
        BATCH)); // batch

    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_C2R, 
        2, // rank
        FFT_Dims,
        CFFT_Dims, 1, odist, // *inembed, istride, idist
        FFT_Dims, 1, idist, // *onembed, ostride, odist
        CUFFT_C2R, 
        BATCH)); // batch
    
    mwSize *FFT_dims = (mwSize *)mxMalloc(2 * sizeof(mwSize));
        FFT_dims[0] = FFT_H;
        FFT_dims[1] = FFT_W;

    /* For each kernel iterate */
    for (int kernelIdx = 0; kernelIdx < N_KERNEL; kernelIdx++){
        
        // Get Kernel Data
        const mxArray *mxCurrentCell = mxGetCell(prhs[KERNLE_CELL_INDEX], kernelIdx);
        if (!mxIsGPUArray(mxCurrentCell)){
            
            if( mxGetClassID(mxCurrentCell) != mxSINGLE_CLASS || mxGetNumberOfDimensions(mxCurrentCell) != 3 )
                mexErrMsgIdAndTxt(errId, "Kernels must be of type float and have features larger than 1");

            h_Kernel = (float *)mxGetData(mxCurrentCell);
            mxKernel_Dim = mxGetDimensions(mxCurrentCell);

            // Kernel dimensions
            KERNEL_H = mxKernel_Dim[0];
            KERNEL_W = mxKernel_Dim[1];
            KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);

            CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **)&d_Kernel, KERNEL_SIZE));
            CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice));
            mxKernel = NULL;
        }else{ // Kernel is GPU Array
            mxKernel = mxGPUCreateFromMxArray(mxCurrentCell);

            if ( mxGPUGetClassID(mxKernel) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(mxKernel) != 3 )
                mexErrMsgIdAndTxt(errId, "Kernels must be of type float and have features larger than 1");

            mxKernel_Dim = mxGPUGetDimensions(mxKernel);

            // Kernel dimensions
            KERNEL_H = mxKernel_Dim[0];
            KERNEL_W = mxKernel_Dim[1];
            KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);

            d_Kernel = (float *)mxGPUGetDataReadOnly(mxKernel);
        }

        if(debug) fprintf(stderr,"Kernel size: h=%d, w=%d\n", KERNEL_H, KERNEL_W);

        if (FEATURE_DIM != mxKernel_Dim[2] || KERNEL_W > FFT_W || KERNEL_H > FFT_H ){
            mexErrMsgIdAndTxt(errId, "Kernel and Data must have the same number of features and kernel size should be smaller than data size");
        }

        padData<<<dataBlockGrid3D, threadBlock3D>>>(
            d_PaddedKernel,
            d_Kernel,
            FFT_W,
            FFT_H,
            KERNEL_W,
            KERNEL_H,
            FEATURE_DIM
            );


        CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, d_PaddedKernel, d_CFFT_KERNEL));
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        if(debug) fprintf(stderr,"FFT done\n");

        
        /* Hadamard product, Element-wise multiplication in frequency domain */
        /* If execute the following, second compile of this file create MATLAB error */
        elementwiseProductAndNormalize<<<dataBlockGrid3D, threadBlock3D>>>(
                d_FFTEProd, // out
                d_CFFT_DATA, // in data
                d_CFFT_KERNEL,   // in kernel
                CFFT_H,
                CFFT_W,
                FEATURE_DIM,
                1.0f / (FFT_W * FFT_H)
            );

        CUFFT_SAFE_CALL(cufftExecC2R(FFTplan_C2R, d_FFTEProd, d_IFFTEProd));
        CUDA_SAFE_CALL_NO_SYNC(cudaDeviceSynchronize());

        sumAlongFeatures<<<dataBlockGrid2D, threadBlock2D>>>(
                d_CONVOLUTION,
                d_IFFTEProd,
                FFT_H,
                FFT_W,
                FEATURE_DIM
            );



        convolutionResult = mxCreateNumericArray(2, FFT_dims, mxSINGLE_CLASS, mxREAL);
        h_CONVOLUTION = (float *)mxGetData(convolutionResult);
        CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_CONVOLUTION, d_CONVOLUTION, CONV_SIZE ,cudaMemcpyDeviceToHost));

        mxSetCell(plhs[CONVOLUTION_CELL_INDEX], kernelIdx, convolutionResult);
    }
    // plhs[1] = mxGPUCreateMxArrayOnGPU(mxFFTKernel);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(mxFFTData);
    // mxGPUDestroyGPUArray(mxConvolution);    
    // mxGPUDestroyGPUArray(mxFFTKernel);
    
    cufftDestroy(FFTplan_R2C);
    cufftDestroy(FFTplan_C2R);

    if(mxKernel == NULL) mxGPUDestroyGPUArray(mxKernel);

    cudaFree(d_PaddedKernel);
    cudaFree(d_IFFTEProd);
    cudaFree(d_CONVOLUTION);
    cudaFree(d_CFFT_KERNEL);
    cudaFree(d_FFTEProd);
    
    if(mxKernel == NULL) cudaFree(d_Kernel);

    mxFree(FFT_dims);
}
