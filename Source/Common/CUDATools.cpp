#ifdef CUDA

#include "CUDATools.h"

size_t cudaFreeMemGet()
{
	size_t LTotal, LFree;
	CUDA_SAFECALL(cudaMemGetInfo(&LFree, &LTotal));
	return LFree;
}

void cudaMemInfoDump()
{
	size_t LTotal, LFree;
	CUDA_SAFECALL(cudaMemGetInfo(&LFree, &LTotal));
	char S[30];
	IntToCharBufF(LTotal, S, 30);
	std::cout << "CUDA Memory Info: " << std::endl;
	std::cout << "    Total: " << S << " byte(s)" << std::endl;
	IntToCharBufF(LFree, S, 30);
	std::cout << "    Free:  " << S << " byte(s)" << std::endl;
}

void cudaDevicePropertiesDump()
{
	cudaDeviceProp LProps;
	int LDevice;
	CUDA_SAFECALL(cudaGetDevice(&LDevice));
	CUDA_SAFECALL(cudaGetDeviceProperties(&LProps, LDevice));
	std::cout << "CUDA Device properties (Device " << LDevice << ")" << std::endl;
	std::cout << "    Name:                        " << LProps.name << std::endl;
	std::cout << "    Version:                     " << IntToStrF(LProps.major) << "." << IntToStrF(LProps.minor) << std::endl;
	std::cout << "    totalGlobalMem:              " << IntToStrF(LProps.totalGlobalMem) << std::endl;
	std::cout << "    sharedMemPerBlock:           " << IntToStrF(LProps.sharedMemPerBlock) << std::endl;
	std::cout << "    regsPerBlock:                " << IntToStrF(LProps.regsPerBlock) << std::endl;
	std::cout << "    warpSize:                    " << IntToStrF(LProps.warpSize) << std::endl;
	std::cout << "    memPitch:                    " << IntToStrF(LProps.memPitch) << std::endl;
	std::cout << "    maxThreadsPerBlock:          " << IntToStrF(LProps.maxThreadsPerBlock) << std::endl;
	std::cout << "    maxThreadsDim[3]:            " << 
		IntToStrF(LProps.maxThreadsDim[0]) << " x " << 
		IntToStrF(LProps.maxThreadsDim[1]) << " x " << 
		IntToStrF(LProps.maxThreadsDim[2]) << std::endl;
	std::cout << "    maxGridSize[3]:              " << 
		IntToStrF(LProps.maxGridSize[0]) << " x " << 
		IntToStrF(LProps.maxGridSize[1]) << " x " << 
		IntToStrF(LProps.maxGridSize[2]) << std::endl;
	std::cout << "    clockRate:                   " << IntToStrF(LProps.clockRate) << std::endl;
	std::cout << "    totalConstMem:               " << IntToStrF(LProps.totalConstMem) << std::endl;
	std::cout << "    textureAlignment:            " << IntToStrF(LProps.textureAlignment) << std::endl;
	std::cout << "    texturePitchAlignment:       " << IntToStrF(LProps.texturePitchAlignment) << std::endl;
	std::cout << "    deviceOverlap:               " << BoolToStrYesNo(LProps.deviceOverlap > 0) << std::endl;
	std::cout << "    multiProcessorCount:         " << IntToStrF(LProps.multiProcessorCount) << std::endl;
	std::cout << "    kernelExecTimeoutEnabled:    " << BoolToStrYesNo(LProps.kernelExecTimeoutEnabled > 0) << std::endl;
	std::cout << "    integrated:                  " << BoolToStrYesNo(LProps.integrated > 0) << std::endl;
	std::cout << "    canMapHostMemory:            " << BoolToStrYesNo(LProps.canMapHostMemory > 0) << std::endl;
	std::cout << "    computeMode:                 " << IntToStrF(LProps.computeMode) << std::endl;
	std::cout << "    concurrentKernels:           " << BoolToStrYesNo(LProps.concurrentKernels > 0) << std::endl;
	std::cout << "    ECCEnabled:                  " << BoolToStrYesNo(LProps.ECCEnabled > 0) << std::endl;
	std::cout << "    pciBusID:                    " << IntToStrF(LProps.pciBusID) << std::endl;
	std::cout << "    pciDeviceID:                 " << IntToStrF(LProps.pciDeviceID) << std::endl;
	std::cout << "    pciDomainID:                 " << IntToStrF(LProps.pciDomainID) << std::endl;
	std::cout << "    tccDriver:                   " << IntToStrF(LProps.tccDriver) << std::endl;
	std::cout << "    asyncEngineCount:            " << IntToStrF(LProps.asyncEngineCount) << std::endl;
	std::cout << "    unifiedAddressing:           " << BoolToStrYesNo(LProps.unifiedAddressing > 0) << std::endl;
	std::cout << "    memoryClockRate:             " << IntToStrF(LProps.memoryClockRate) << std::endl;
	std::cout << "    memoryBusWidth:              " << IntToStrF(LProps.memoryBusWidth) << std::endl;
	std::cout << "    L2CacheSize:                 " << IntToStrF(LProps.l2CacheSize) << std::endl;
	std::cout << "    maxThreadsPerMultiProcessor: " << IntToStrF(LProps.maxThreadsPerMultiProcessor) << std::endl;
}

void cudaSafeCall(cudaError error, const char *flp, int line)
{
    if (error != cudaSuccess)
		throw new ECUDAException(cudaGetErrorString(error), error, flp, line);
}

void cufftSafeCall(cufftResult result, const char *flp, int line)
{
    if (result != CUFFT_SUCCESS)
		throw new ECUFFTException("cuFFT call failed", result, flp, line);
}

void curandSafeCall(curandStatus_t status, const char *flp, int line)
{
    if (status != CURAND_STATUS_SUCCESS)
		throw new ECURANDException("cuRAND call failed", status, flp, line);
}

#endif