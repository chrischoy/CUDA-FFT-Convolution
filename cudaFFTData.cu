/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */


#include <cuda.h>
#include <cufft.h>
#include "cutil.h"
#include "mex.h"
#include "gpu/mxGPUArray.h"
// #include "convolutionFFTkernel.cu"

typedef float2 Complex;
static bool debug = true;

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
    const mxArray *mxDATA = prhs[0];
    mxGPUArray *FFT_DATA;
    float2 *d_CFFT_DATA;
    float *h_Data;
    float *d_Data;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    // int const THREAD_PER_BLOCK_X = 16;
    // int const THREAD_PER_BLOCK_Y = 8;
    // int const THREAD_PER_BLOCK_Z = 32;
    // int MblocksPerGrid, NblocksPerGrid;
    int KERNEL_H, KERNEL_W, DATA_H, DATA_W, PADDING_H, PADDING_W, FFT_H, FFT_W, FEATURE_DIM;
    int DATA_SIZE, FFT_SIZE, CFFT_SIZE;

    
    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    
    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=3) || (mxIsGPUArray(mxDATA)) || mxGetNumberOfDimensions(mxDATA) != 3 || mxGetClassID(mxDATA) != mxSINGLE_CLASS) {
        mexErrMsgIdAndTxt(errId, errMsg);
    }


    // Kernel dimensions
    KERNEL_H = (int)mxGetScalar(prhs[1]);
    KERNEL_W = (int)mxGetScalar(prhs[2]);
    if(debug) fprintf(stderr,"Kernel size: h=%d, w=%d\n",KERNEL_H,KERNEL_W);

    // Data dimensions
    const mwSize *DATA_dims = mxGetDimensions(mxDATA);
    DATA_H = DATA_dims[0];
    DATA_W = DATA_dims[1];
    FEATURE_DIM = DATA_dims[2];
    if(debug) fprintf(stderr,"Data size: h=%d, w=%d\n",DATA_H,DATA_W);

    // Width and height of padding
    PADDING_H = KERNEL_H - 1;
    PADDING_W = KERNEL_W - 1;

    // Derive FFT size from data and kernel dimensions
    // fprintf(stderr,"Calculating FFT size\n");
    FFT_H = computeFFTsize(DATA_H + PADDING_H);
    FFT_W = computeFFTsize(DATA_W + PADDING_W);
    if(debug) fprintf(stderr,"FFT size: h=%d, w=%d\n",FFT_H,FFT_W);

    DATA_SIZE = DATA_W * DATA_H * FEATURE_DIM * sizeof(float);
    FFT_SIZE  = FFT_W * FFT_H * FEATURE_DIM * sizeof(float);
    CFFT_SIZE = FFT_W * FFT_H * FEATURE_DIM * sizeof(Complex);

    // Allocate memory for input
    // No need to initialize using mxCalloc
    h_Data = (float *)mxGetData(mxDATA);

    mwSize *FFT_dims = (mwSize *)mxMalloc(3 * sizeof(mwSize));

    FFT_dims[0] = FFT_H;
    FFT_dims[1] = FFT_W;
    FFT_dims[2] = FEATURE_DIM;

    /* Wrap the result up as a MATLAB gpuArray for return. */
    FFT_DATA = mxGPUCreateGPUArray(3,
                                FFT_dims,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_INITIALIZE_VALUES);
    
    d_CFFT_DATA = (Complex *)mxGPUGetData(FFT_DATA);
    
    cudaMalloc((void **)&d_Data,        DATA_SIZE);
    // cudaMalloc((void **)&d_PaddedData,  FFT_SIZE);
    // cudaMalloc((void **)&d_CFFT_DATA,   CFFT_SIZE);

    // cudaMemset(d_PaddedData,   0, FFT_SIZE);
    // cufftPlan2d(&FFTplan_R2C, FFT_H, FFT_W, CUFFT_R2C);

    int BATCH = FEATURE_DIM;
    
    int RANK = 2;
    
    int FFT_Dims[2];
    FFT_Dims[0] = FFT_H;
    FFT_Dims[1] = FFT_W;
    
    int inembed [2];
    inembed[0] = DATA_H;
    inembed[1] = DATA_W;

    int istrid = 1;
    int idist = DATA_W * DATA_H;

    int onembed [2];
    inembed[0] = FFT_H;
    inembed[1] = FFT_W;

    int ostrid = 1;
    int odist = FFT_W * FFT_H;
    
    cufftHandle FFTplan_R2C;
    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_R2C, RANK, FFT_Dims, inembed, istrid, idist, onembed, ostrid, odist, CUFFT_R2C, BATCH));

    cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice);

    CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, (cufftReal *)d_Data, (cufftComplex *)d_CFFT_DATA));
    cudaDeviceSynchronize();

    plhs[0] = mxGPUCreateMxArrayOnGPU(FFT_DATA);

    cufftDestroy(FFTplan_R2C);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(FFT_DATA);
    cudaFree(d_Data);
    // mxGPUDestroyGPUArray(d_CFFT_DATA);
}
