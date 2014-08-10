#ifndef CUDA_CONV_FFT_DATA
#define CUDA_CONV_FFT_DATA

#  define IMUL(a, b) __mul24(a, b)

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i .\n",            \
                __FILE__, __LINE__);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i .\n",            \
                __FILE__, __LINE__ );                                        \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUFFT_SAFE_CALL( call) do {                                        \
    cufftResult err = call;                                                  \
    if( CUFFT_SUCCESS != err) {                                              \
        fprintf(stderr, "CUFFT error in file '%s' in line %i.\n",            \
                __FILE__, __LINE__);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


typedef struct
{
    /* Concurrency parameters */
    int GPU_IDX, STREAM_IDX;

    mxArray *convolutionResult;
    
    /* cufftComplex is float2 */
    const cufftComplex *d_CFFT_DATA;
    cufftComplex *d_CFFT_KERNEL;

    float *d_CONVOLUTION;
    float *d_IFFTEProd;

    float *h_Kernel;
    float *h_CONVOLUTION;
    float *d_Kernel;
    float *d_PaddedKernel;

    //Stream for asynchronous command execution
    cudaStream_t stream;

} ConvPlan;

#endif