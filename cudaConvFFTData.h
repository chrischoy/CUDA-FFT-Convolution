#ifndef CUDA_CONV_FFT_DATA
#define CUDA_CONV_FFT_DATA

#  define IMUL(a, b) __mul24(a, b)

#  define CUDA_SAFE_CALL_NO_SYNC( call) do {                                 \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__, err);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUDA_SAFE_CALL( call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError err = cudaThreadSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        printf("Cuda error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__,err);                                        \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#  define CUFFT_SAFE_CALL( call) do {                                        \
    cufftResult err = call;                                                  \
    if( CUFFT_SUCCESS != err) {                                              \
        printf("CUFFT error in file '%s' in line %i Error : %d.\n",            \
                __FILE__, __LINE__,err);                                         \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)


typedef struct
{
    /* Concurrency parameters */
    int GPU_IDX, STREAM_IDX;
    cufftHandle FFTplan_R2C, FFTplan_C2R;
    const cufftComplex *d_CFFT_DATA;

    mxArray *convolutionResult;

    cufftComplex *d_CFFT_KERNEL;
    cufftComplex *d_FFTEProd;
    float *d_CONVOLUTION;
    float *d_IFFTEProd;
    float *d_Kernel;
    float *d_PaddedKernel;

    float *h_Kernel;
    float *h_CONVOLUTION;

    const mwSize *mxKernel_Dim;

    //Stream for asynchronous command execution
    cudaStream_t stream;

} ConvPlan;

int checkDeviceProp ( cudaDeviceProp p ) {
    int support = p.canMapHostMemory;

    if(support == 0) printf( "%s does not support mapping host memory.\n", p.name);
    else             printf( "%s supports mapping host memory.\n",p.name);

    support = p.concurrentKernels;
    if(support == 0) printf("%s does not support concurrent kernels\n", p.name);
    else printf("%s supports concurrent kernels\n",p.name);

    support = p.kernelExecTimeoutEnabled;
    if(support == 0) printf("%s kernelExecTimeout disabled\n", p.name);
    else printf("%s kernelExecTimeout enabled\n",p.name);

    printf("compute capability : %d.%d \n", p.major,p.minor);
    printf("number of multiprocessors : %d \n", p.multiProcessorCount);

    return support;
}

#endif