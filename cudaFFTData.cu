/*
 * Example of how to use the mxGPUArray API in a MEX file.  This example shows
 * how to write a MEX function that takes a gpuArray input and returns a
 * gpuArray output, e.g. B=mexFunction(A).
 *
 * Copyright 2012 The MathWorks, Inc.
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"

typedef float2 Complex;
static bool debug = true;

/*
 * Device code
 */
void __global__ TimesTwo(double const * const A,
                         double * const B,
                         int const N)
{
    /* Calculate the global linear index, assuming a 1-d grid. */
    int const i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        B[i] = 2.0 * A[i];
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
    const mxArray *mxDATA = prhs[0];
    mxGPUArray *FFT_DATA;
    float2 *d_CFFT_DATA;
    float *h_Data;
    int N;
    char const * const errId = "parallel:gpu:mexGPUExample:InvalidInput";
    char const * const errMsg = "Invalid input to MEX file.";

    /* Choose a reasonably sized number of threads for the block. */
    int const threadsPerBlock = 32;
    int MblocksPerGrid, NblocksPerGrid;
    int KERNEL_H, KERNEL_W, DATA_H, DATA_W, PADDING_H, PADDING_W, FF_H, FFT_W;

    cufftHandle FFTplan_R2C;
    cufftHandle FFTplan_C2R;

    /* Initialize the MathWorks GPU API. */
    mxInitGPU();

    /* Throw an error if the input is not a GPU array. */
    if ((nrhs!=3) || (mxIsGPUArray(mxDATA)) || mxGetNumberOfDimensions(mxDATA) != 3) {
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
    plhs[0] = mxGPUCreateGPUArray(3,
                                FFT_dims,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_INITIALIZE_VALUES);
    
    d_CFFT_DATA = (Complex *)mxGPUGetData(plhr[0]);
    
    cudaMalloc((void **)&d_Data,            DATA_SIZE);
    cudaMalloc((void **)&d_PaddedData,      FFT_SIZE) ;
    cudaMalloc((void **)&fft_PaddedData,    CFFT_SIZE);

    cufftPlan2d(&FFTplan_R2C, FFT_H, FFT_W, CUFFT_R2C) ;

    /*
     * Call the kernel using the CUDA runtime API. We are using a 1-d grid here,
     * and it would be possible for the number of elements to be too large for
     * the grid. For this example we are not guarding against this possibility.
     */
    N = (int)(mxGPUGetNumberOfElements(A));
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    TimesTwo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, N);

    plhs[0] = mxGPUCreateMxArrayOnGPU(B);

    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(A);
    mxGPUDestroyGPUArray(B);
}
