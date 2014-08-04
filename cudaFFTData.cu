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

int computeFFTsize16(int dataSize){
    // Compute the multiple of 16
    int mod = dataSize / 16;
    int rem = dataSize % 16;

    return (mod * 16) + ((rem > 0)?16:0);
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
    float *d_PaddedData;
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
    if ((nrhs!=3) ||
            mxIsGPUArray(mxDATA) || 
            mxGetNumberOfDimensions(mxDATA) != 3 || 
            mxGetClassID(mxDATA) != mxSINGLE_CLASS) {
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

    h_Data = (float *)mxGetData(mxDATA);
    if(debug) fprintf(stderr,"Data size: h=%d, w=%d, f=%d\n",DATA_H,DATA_W,FEATURE_DIM); 

    // Width and height of padding
    PADDING_H = KERNEL_H - 1;
    PADDING_W = KERNEL_W - 1;

    // Derive FFT size from data and kernel dimensions
    // FFT_H = computeFFTsize(DATA_H + PADDING_H);
    // FFT_W = computeFFTsize(DATA_W + PADDING_W);
    FFT_H = computeFFTsize16(DATA_H + PADDING_H);
    FFT_W = computeFFTsize16(DATA_W + PADDING_W);

    if(debug) fprintf(stderr,"FFT size: h=%d, w=%d\n",FFT_H,FFT_W);

    DATA_SIZE = DATA_W * DATA_H * FEATURE_DIM * sizeof(float);
    FFT_SIZE  = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float);
    CFFT_SIZE = FFT_W  * FFT_H  * FEATURE_DIM * sizeof(float2);

    // Allocate memory for input
    // No need to initialize using mxCalloc
    
    mwSize *FFT_dims = (mwSize *)mxMalloc(3 * sizeof(mwSize));

    FFT_dims[0] = FFT_H/2 + 1;
    FFT_dims[1] = FFT_W;
    FFT_dims[2] = FEATURE_DIM;

    /* Wrap the result up as a MATLAB gpuArray for return. */
    FFT_DATA = mxGPUCreateGPUArray(3,
                                FFT_dims,
                                mxSINGLE_CLASS,
                                mxCOMPLEX,
                                MX_GPU_INITIALIZE_VALUES);
    
    d_CFFT_DATA = (float2 *)mxGPUGetData(FFT_DATA);
    
    cudaMalloc((void **)&d_Data,        DATA_SIZE);
    cudaMalloc((void **)&d_PaddedData,  FFT_SIZE);

    cudaMemcpy(d_Data, h_Data, DATA_SIZE, cudaMemcpyHostToDevice);

    dim3 threadBlock(THREAD_PER_BLOCK_H, THREAD_PER_BLOCK_W, THREAD_PER_BLOCK_D);
    dim3 dataBlockGrid( iDivUp(FFT_W, threadBlock.x), 
                        iDivUp(FFT_H, threadBlock.y), 
                        iDivUp(FEATURE_DIM, threadBlock.z));

    padData<<<dataBlockGrid, threadBlock>>>(
        d_PaddedData,
        d_Data,
        FFT_W,
        FFT_H,
        DATA_W,
        DATA_H,
        FEATURE_DIM
        );

    if(debug) fprintf(stderr,"Padding\n");

    int BATCH = FEATURE_DIM;
    int FFT_Dims[] = { FFT_W, FFT_H };

    int idist = FFT_W * FFT_H;
    int odist = FFT_W * (FFT_H/2 + 1);
    
    int inembed[] = {FFT_W, FFT_H};
    int onembed[] = {FFT_W, FFT_H/2 + 1};

    cufftHandle FFTplan_R2C;
    CUFFT_SAFE_CALL(cufftPlanMany(&FFTplan_R2C, 
        2, // rank
        FFT_Dims, 
        inembed, 1, idist, // *inembed, istride, idist
        onembed, 1, odist, // *onembed, ostride, odist
        CUFFT_R2C, 
        BATCH)); // batch


    CUFFT_SAFE_CALL(cufftExecR2C(FFTplan_R2C, d_PaddedData, d_CFFT_DATA));
    CUFFT_SAFE_CALL(cudaDeviceSynchronize());
    if(debug) fprintf(stderr,"Sync\n");

    plhs[0] = mxGPUCreateMxArrayOnGPU(FFT_DATA);
    if(debug) fprintf(stderr,"plhs\n");
    /*
     * The mxGPUArray pointers are host-side structures that refer to device
     * data. These must be destroyed before leaving the MEX function.
     */
    mxGPUDestroyGPUArray(FFT_DATA);
    cufftDestroy(FFTplan_R2C);
    cudaFree(d_Data);
    cudaFree(d_PaddedData);
    mxFree(FFT_dims);
}
