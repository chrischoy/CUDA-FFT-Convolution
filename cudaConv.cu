#include <cuda.h>
#include <cufft.h>
#include "cutil.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"

#define IMUL(a, b) __mul24(a, b)
static bool debug = true;

/*
 * Device Code
 */
__global__ void padData(
    float *d_PaddedData,
    float *d_Data,
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
// Data configuration
////////////////////////////////////////////////////////////////////////////////
int computeFFTsize(int dataSize){
    //Highest non-zero bit position of dataSize
    int hiBit;
    //Neares lower and higher powers of two numbers for dataSize
    unsigned int lowPOT, hiPOT;

    //Align data size to a multiple of half-warp
    //in order to have each line starting at properly aligned addresses
    //for coalesced global memory writes in padKernel() and padData()
    dataSize = iAlignUp(dataSize, 16);

    //Find highest non-zero bit
    for(hiBit = 31; hiBit >= 0; hiBit--)
        if(dataSize & (1U << hiBit)) break;

    //No need to align, if already power of two
    lowPOT = 1U << hiBit;
    if(lowPOT == dataSize) return dataSize;

    //Align to a nearest higher power of two, if the size is small enough,
    //else align only to a nearest higher multiple of 512,
    //in order to save computation and memory bandwidth
    hiPOT = 1U << (hiBit + 1);
    //if(hiPOT <= 1024)
        return hiPOT;
    //else 
    //  return iAlignUp(dataSize, 512);
}


////////////////////////////////////////////////////////////////////////////////
// Mex Entry
////////////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, mxArray const *prhs[])
{
    /* Declare all variables.*/
    const mxArray *mxFFTData = prhs[0];
    const mxArray *mxKernel = prhs[1];
    mxGPUArray *mxFFTKernel;
    float2 *d_CFFT_DATA;
    float2 *d_CFFT_KERNEL;

    float *h_Kernel;
    float *d_Kernel;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    int const THREAD_PER_BLOCK_H = 16;
    int const THREAD_PER_BLOCK_W = 8;
    int const THREAD_PER_BLOCK_D = 8;

    // int MblocksPerGrid, NblocksPerGrid;
    int KERNEL_H, KERNEL_W, DATA_H, DATA_W, 
        PADDING_H, PADDING_W, FFT_H, FFT_W, FEATURE_DIM,
        DATA_SIZE, FFT_SIZE, CFFT_SIZE;

    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=2) ||
            !mxIsGPUArray(mxFFTData) || 
            mxGetNumberOfDimensions(mxFFTData) != 3 || 
            mxGetClassID(mxFFTData) != mxSINGLE_CLASS ||
            mxGetClassID(mxKernel) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    // FFT Dim
    mwSize const * mxFFT_Dim = mxGPUGetDimensions(mxFFTData);
    FFT_H = mxFFT_Dim[0];
    FFT_W = mxFFT_Dim[1];
    FEATURE_DIM = mxFFT_Dim[2];
    if(debug) fprintf(stderr,"FFT Data size: h=%d, w=%d\n", FFT_H, FFT_W);

    // Kernel dimensions
    const mwSize * Kernel_Dim = mxGetDimensions(mxKernel);
    KERNEL_H = Kernel_Dim[0];
    KERNEL_W = Kernel_Dim[1];

    if (FEATURE_DIM != Kernel_Dim[2] || KERNEL_W > FFT_W || KERNEL_H > FFT_H ){
        mexErrMsgIdAndTxt(errId, errMsg);
    }

    if(debug) fprintf(stderr,"Kernel size: h=%d, w=%d\n", KERNEL_H, KERNEL_W);

    KERNEL_SIZE = KERNEL_W * KERNEL_H * FEATURE_DIM * sizeof(float);
    FFT_SIZE  = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float);
    CFFT_SIZE = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float2);

    // Get Complex FFT data handle
    d_CFFT_DATA = (float2 *)mxGPUGetData(mxFFTData);

    // Get Kernel Data
    if (!mxIsGPUArray(mxKernel)){
        h_Kernel = (float *)mxGetData(mxKernel);
        cudaMalloc((void **)&d_Kernel, KERNEL_SIZE);
        cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE, cudaMemcpyHostToDevice);
    }else{
        d_Kernel = (float *)mxGPUGetData(mxKernel);
    }

    /*  Pad Kernel */
    /* Create a GPUArray to hold the result and get its underlying pointer. */
    mxFFTKernel = mxGPUCreateGPUArray(3,
                            mxFFT_Dim,
                            mxSINGLE_CLASS,
                            mxCOMPLEX,
                            MX_GPU_DO_NOT_INITIALIZE);
    d_CFFT_KERNEL = (double *)(mxGPUGetData(mxFFTKernel));

    dim3 threadBlock(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    dim3 dataBlockGrid( iDivUp(FFT_W, threadBlock.x), 
                        iDivUp(FFT_H, threadBlock.y), 
                        iDivUp(FEATURE_DIM, threadBlock.z));

    padData<<<dataBlockGrid, threadBlock>>>(
        (float *)d_CFFT_KERNEL,
        d_Kernel,
        FFT_W,
        FFT_H,
        KERNEL_W,
        KERNEL_H,
        FEATURE_DIM
        );

    int FFT_Dims[] = { FFT_W, FFT_H };
    int dist = FFT_H * FFT_W;
    
    cufftHandle FFTplan_R2C;
    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_R2C, 
        2, // rank
        FFT_Dims, 
        FFT_Dims, 1, dist, // *inembed, istride, idist
        FFT_Dims, 1, dist, // *onembed, ostride, odist
        CUFFT_R2C, 
        FEATURE_DIM)); // batch

    CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, (float *)d_CFFT_KERNEL, d_CFFT_KERNEL));
    CUFFT_SAFE_CALL(cudaDeviceSynchronize());

    plhs[0] = mxGPUCreateMxArrayOnGPU(mxFFTKernel);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(mxFFTKernel);
    cufftDestroy(FFTplan_R2C);
}
