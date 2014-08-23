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