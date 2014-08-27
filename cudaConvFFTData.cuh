#ifndef CUDA_CONV_FFT_DATA_CUH
#define CUDA_CONV_FFT_DATA_CUH

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
If a variable located in global or shared memory is declared as volatile, 
the compiler assumes that its value can be changed or used at any time by 
another thread and therefore any reference to this variable compiles to an
 actual memory read or write instruction.
*/
__global__ void sumAlongFeaturesReduction(
    float *convolutionResult,
    const volatile float *convolutionPerFeature,
    int FFT_H,
    int FFT_W,
    int FEATURE_DIM,
    int PREV_POW_2_FEATURE_DIM
){
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;

    
    if(x < FFT_W && y < FFT_H && z < PREV_POW_2_FEATURE_DIM){
        const int result_i = IMUL(FFT_H, x) + y;
        const int N = IMUL(FFT_W, FFT_H);
        uint idx = PREV_POW_2_FEATURE_DIM;
        while(idx >= 1){
            idx = idx >> 1;
            convolutionPerFeature[result_i + IMUL(z, N)] += (z+idx < FEATURE_DIM)?convolutionPerFeature[result_i + IMUL(z + idx, N)]:0;
            __synchronizeThread();
        }
        convolutionResult[result_i] = convolutionPerFeature[result_i];
    }
}


__device__ int getGlobalIdx_2D_Grid_3D_Block()
{
    int blockId = blockIdx.x 
             + blockIdx.y * gridDim.x; 
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
               + (threadIdx.z * (blockDim.x * blockDim.y))
               + (threadIdx.y * blockDim.x)
               + threadIdx.x;
    return threadId;
} 

/* Unrolled version of reduction */
__global__ void sumAlongFeaturesReductionUnroll32(
    float *convolutionResult,
    const volatile float *convolutionPerFeature,
    int FFT_H,
    int FFT_W,
    int N,
    int FEATURE_DIM
){
    extern __shared__ int sdata[];
    const int x = IMUL(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = IMUL(blockDim.y, blockIdx.y) + threadIdx.y;
    const int z = IMUL(blockDim.z, blockIdx.z) + threadIdx.z;
    
    if(x < FFT_W && y < FFT_H && z < 16){
        const unsigned int SharedN = IMUL(blockDim.x, blockDim.y);
        const unsigned int xyTid = IMUL(threadIdx.y, blockDim.x)
                                    + threadIdx.x;
        const int tid       = IMUL(threadIdx.z, SharedN) + xyTid;
        const int result_i  = IMUL(FFT_H, x) + y;

        sdata[tid] = convolutionPerFeature[result_i + IMUL(z, N)] + (z + 16 < FEATURE_DIM) ? convolutionPerFeature[result_i + IMUL(z + 16, N)] : 0;
        __synchronizeThread();
        sdata[tid] +=sdata[tid + IMUL( 8, SharedN)];
        sdata[tid] +=sdata[tid + IMUL( 4, SharedN)];
        sdata[tid] +=sdata[tid + IMUL( 2, SharedN)];
        sdata[tid] +=sdata[tid + SharedN];
        if (threadIdx.z == 0){
            convolutionResult[result_i] = sdata[xyTid];
        }
    }
}


/* Instructions are SIMD synchronous within a warp */
template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >=  8) sdata[tid] += sdata[tid + 4];
    if (blockSize >=  4) sdata[tid] += sdata[tid + 2];
    if (blockSize >=  2) sdata[tid] += sdata[tid + 1];
}

// unsigned int sharedSize = numThreads*sizeof(int);
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata, unsigned int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid; 
    unsigned int gridSize = blockSize*2*gridDim.x; sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += g_idata[i] + g_idata[i+blockSize]; 
        i += gridSize; 
    }
    __syncthreads();
    if (blockSize >= 512) { 
        if (tid < 256) sdata[tid] += sdata[tid + 256];
        __syncthreads(); 
    } 
    if (blockSize >= 256) { 
        if (tid < 128) sdata[tid] += sdata[tid + 128]; 
        __syncthreads(); 
    } 
    if (blockSize >= 128) { 
        if (tid < 64) sdata[tid] += sdata[tid + 64]; 
        __syncthreads();
    }
    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0]; 
}    

#endif