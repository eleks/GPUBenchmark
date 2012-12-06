#ifdef CUDA

#include <cuda_runtime_api.h>
#include "Common/CUDATools.h"
#include "GPUBenchmarkCUDA.h"

void GPUBenchmarkCUDA::deviceInitialize()
{
	int deviceCount;
	CUDA_SAFECALL(cudaGetDeviceCount(&deviceCount));
	if (deviceCount > 1)
		CUDA_SAFECALL(cudaSetDevice(1));
}

void GPUBenchmarkCUDA::devicePropertiesShow()
{
	int device;
	cudaDeviceProp deviceProps;
	CUDA_SAFECALL(cudaGetDevice(&device));
	CUDA_SAFECALL(cudaGetDeviceProperties(&deviceProps, device));
	deviceClockRate = deviceProps.clockRate * 1000;
	
	printf("\n");
	printf("GPU properties\n");
	printf("Name:                           %30s\n", deviceProps.name);
	printf("Clock rate:                     %30s\n", IntToStrF(deviceClockRate).c_str());
	printf("Memory clock rate:              %30s\n", IntToStrF((__int64)deviceProps.memoryClockRate * 1000).c_str());
	printf("Multiprocessors:                %30d\n", deviceProps.multiProcessorCount);
	printf("Maximum resident threads per multiprocessor:        %10d\n", deviceProps.maxThreadsPerMultiProcessor);

	printf("Version (compute capability):   %30s\n", (IntToStrF(deviceProps.major) + "." + IntToStrF(deviceProps.minor)).c_str());
	printf("Total global memory:            %30s\n", IntToStrF(deviceProps.totalGlobalMem).c_str());
	printf("Shared memory per Block:        %30s\n", IntToStrF(deviceProps.sharedMemPerBlock).c_str());
	printf("Registers per Block:            %30s\n", IntToStrF(deviceProps.regsPerBlock).c_str());
	printf("Warp size:                      %30s\n", IntToStrF(deviceProps.warpSize).c_str());
	printf("Mem pitch:                      %30s\n", IntToStrF(deviceProps.memPitch).c_str());
	printf("Max threads per block:          %30s\n", IntToStrF(deviceProps.maxThreadsPerBlock).c_str());
	printf("Max threads dimentions:    %35s\n", 
		(IntToStrF(deviceProps.maxThreadsDim[0]) + " x " + 
		IntToStrF(deviceProps.maxThreadsDim[1]) + " x " + 
		IntToStrF(deviceProps.maxThreadsDim[2])).c_str());
	printf("Max grid size:             %35s\n", 
		(IntToStrF(deviceProps.maxGridSize[0]) + " x " + 
		IntToStrF(deviceProps.maxGridSize[1]) + " x " + 
		IntToStrF(deviceProps.maxGridSize[2])).c_str());
	printf("Total const memory:             %30s\n", IntToStrF(deviceProps.totalConstMem).c_str());
	printf("Texture alignment:              %30s\n", IntToStrF(deviceProps.textureAlignment).c_str());
	printf("Texture pitch alignment:        %30s\n", IntToStrF(deviceProps.texturePitchAlignment).c_str());
	printf("Device overlap:                 %30s\n", BoolToStrYesNo(deviceProps.deviceOverlap > 0).c_str());
	printf("Kernel exec timeout enabled:    %30s\n", BoolToStrYesNo(deviceProps.kernelExecTimeoutEnabled > 0).c_str());
	printf("Integrated:                     %30s\n", BoolToStrYesNo(deviceProps.integrated > 0).c_str());
	printf("Can map host memory:            %30s\n", BoolToStrYesNo(deviceProps.canMapHostMemory > 0).c_str());
	printf("Compute mode:                   %30s\n", IntToStrF(deviceProps.computeMode).c_str());
	printf("Concurrent kernels:             %30s\n", BoolToStrYesNo(deviceProps.concurrentKernels > 0).c_str());
	printf("ECC enabled:                    %30s\n", BoolToStrYesNo(deviceProps.ECCEnabled > 0).c_str());
	printf("PCI bus ID:                     %30s\n", IntToStrF(deviceProps.pciBusID).c_str());
	printf("PCI device ID:                  %30s\n", IntToStrF(deviceProps.pciDeviceID).c_str());
	printf("PCI domain ID:                  %30s\n", IntToStrF(deviceProps.pciDomainID).c_str());
	printf("TCC driver:                     %30s\n", IntToStrF(deviceProps.tccDriver).c_str());
	printf("Async engine count:             %30s\n", IntToStrF(deviceProps.asyncEngineCount).c_str());
	printf("Unified addressing:             %30s\n", BoolToStrYesNo(deviceProps.unifiedAddressing > 0).c_str());
	printf("Memory bus width:               %30s\n", IntToStrF(deviceProps.memoryBusWidth).c_str());
	printf("L2 cache size:                  %30s\n", IntToStrF(deviceProps.l2CacheSize).c_str());
	printf("\n");
}

void GPUBenchmarkCUDA::deviceMemAllocRelease(int size, int repeatCount, int, int, int)
{
	deviceMem<byte> dmem;
	for (int i = 0; i < repeatCount; i++)
	{
		dmem.allocate(size);
		dmem.release();
	}
}

void GPUBenchmarkCUDA::mappedMemAllocRelease(int size, int repeatCount, int, int, int)
{
	mappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.allocate(size);
		mmem.release();
	}
}

void GPUBenchmarkCUDA::hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int)
{
	hostMem<byte> hmem;
	for (int i = 0; i < repeatCount; i++)
	{
		hmem.allocate(size, cudaHostAllocWriteCombined);
		hmem.release();
	}
}

void GPUBenchmarkCUDA::hostMemRegisterUnregister(int size, int repeatCount, int, int, int)
{
	hostMem<byte> hmem(size);
	mappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.registerHost(hmem.hptr(), size);
		mmem.release();
	}
}
/*
double GPUBenchmarkCUDA::hostMemTransfer(int size, int streamCount, int mode, int direction, int)
{
	streamSize = size / streamCount;
	hostMem<deviceMem<byte> > dmem(streamCount);
	hostMem<deviceMem<byte> > dmem2(streamCount);
	hostMem<hostMem<byte> > hmem(streamCount);
	hostMem<cudaStream_t> streams(streamCount);
	for (int i = 0; i < streamCount; i++)
	{
		dmem[i].allocate(streamSize);
		if (direction == 2)
			dmem2[i].allocate(streamSize);
		switch (mode)
		{
		case 0:
			hmem[i].allocate(streamSize);
			break;
		case 1:
			hmem[i].allocate(streamSize, cudaHostAllocDefault);
			break;
		case 2:
			hmem[i].allocate(streamSize, cudaHostAllocWriteCombined);
			break;
		default:
			assert(false);
		}
		hmem[i].copyTo(dmem[i]); // device memory initialization
		CUDA_SAFECALL(cudaStreamCreate(&streams[i]));
	}
	TimingCounter counter;
	TimingClearAndStart(counter);
	for (int i = 0; i < streamCount; i++)
		switch (direction)
		{
		case 0:
			hmem[i].copyToAsync(dmem[i], streams[i]);
			break;
		case 1:
			dmem[i].copyToAsync(hmem[i], streams[i]);
			break;
		case 2:
			dmem[i].copyToAsync(dmem2[i], streams[i]);
			break;
		default:
			assert(false);
		}
	CUDA_SAFECALL(cudaDeviceSynchronize());
	TimingFinish(counter);

	for (int i = 0; i < streamCount; i++)
		CUDA_SAFECALL(cudaStreamDestroy(streams[i]));

	return TimingSeconds(counter);
}
*/
double GPUBenchmarkCUDA::memTransfer(int size, int mode, int direction, int, int)
{
	const int ITERATIONS = 10;
	deviceMem<byte> dmem;
	deviceMem<byte> dmem2;
	hostMem<byte> hmem;
	dmem.allocate(size);
	if (direction == 2)
		dmem2.allocate(size);
	switch (mode)
	{
	case 0:
		hmem.allocate(size);
		break;
	case 1:
		hmem.allocate(size, cudaHostAllocDefault);
		break;
	case 2:
		hmem.allocate(size, cudaHostAllocWriteCombined);
		break;
	default:
		assert(false);
	}
	hmem.copyTo(dmem); // device memory initialization
	TimingCounter counter;
	TimingClearAndStart(counter);
	for (int i = 0; i < ITERATIONS; i++)
		switch (direction)
		{
		case 0:
			hmem.copyTo(dmem);
			break;
		case 1:
			dmem.copyTo(hmem);
			break;
		case 2:
			dmem.copyTo(dmem2);
			break;
		default:
			assert(false);
		}
	CUDA_SAFECALL(cudaDeviceSynchronize());
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmarkCUDA::kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
	{
		cuda_doTinyTask(blockCount, threadCount);
		CUDA_SAFECALL(cudaDeviceSynchronize());
	}
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmarkCUDA::kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
		cuda_doTinyTask(blockCount, threadCount);
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

void GPUBenchmarkCUDA::kernelAdd_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_int64(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelAdd(int count, int blockCount, int threadCount, int, int)
{
	cuda_doAdd<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelAdd_indep_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd_indep<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_indep_int64(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd_indep<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_indep_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd_indep<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelAdd_indep_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd_indep<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelAdd_indep(int count, int blockCount, int threadCount, int, int)
{
	cuda_doAdd_indep<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelMad_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMad<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMad_int64(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMad<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMad_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMad<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMad_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMad<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelMad(int count, int blockCount, int threadCount, int, int)
{
	cuda_doMad<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelMadSF_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMadSF<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMadSF_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMadSF<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelMadSF(int count, int blockCount, int threadCount, int, int)
{
	cuda_doMadSF<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelMul_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMul<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMul_int64(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMul<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMul_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMul<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelMul_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelMul<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelMul(int count, int blockCount, int threadCount, int, int)
{
	cuda_doMul<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelDiv_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelDiv<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelDiv_int64(int count, int blockCount, int threadCount, int, int) 
{
	return kernelDiv<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelDiv_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelDiv<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelDiv_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelDiv<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelDiv(int count, int blockCount, int threadCount, int, int)
{
	cuda_doDiv<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

void GPUBenchmarkCUDA::kernelSin_float(int count, int blockCount, int threadCount, int, int) 
{
	return kernelSin<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCUDA::kernelSin_double(int count, int blockCount, int threadCount, int, int) 
{
	return kernelSin<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCUDA::kernelSin(int count, int blockCount, int threadCount, int, int)
{
	cuda_doSin<T>(count, blockCount, threadCount);
	CUDA_SAFECALL(cudaDeviceSynchronize());
}

#ifdef CUDA50
double GPUBenchmarkCUDA::kernelDynamicExecuteTinyTask(int blockCount, int threadCount, int, int, int)
{
	return (cuda_doDynamicTinyTask(blockCount, threadCount, 1) / deviceClockRate);
}

double GPUBenchmarkCUDA::kernelDynamicScheduleTinyTask(int blockCount, int threadCount, int, int, int)
{
	return (cuda_doDynamicTinyTask(blockCount, threadCount, 0) / deviceClockRate);
}
#endif

double GPUBenchmarkCUDA::deviceMemAccess_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess)
{
	return deviceMemAccess<int>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

double GPUBenchmarkCUDA::deviceMemAccess_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) 
{
	return deviceMemAccess<float>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

double GPUBenchmarkCUDA::deviceMemAccess_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) 
{
	return deviceMemAccess<double>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

template<typename T> 
double GPUBenchmarkCUDA::deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess)
{
	TimingCounter counter;
	deviceMem<T> dData;
	mappedMem<T> mData;
	T *dptr;
	switch (memAccess)
	{
	case maPinnedAlignedRead:
	case maPinnedAlignedWrite:
	case maPinnedNotAlignedRead:
	case maPinnedNotAlignedWrite:
		mData.allocate(count);
		memset(mData.hptr(), 0, count * sizeof(T));
		dptr = mData.dptr();
		break;
	default:
		dData.allocate(count);
		dData.clear();
		dptr = dData.dptr();
	}
	CUDA_SAFECALL(cudaDeviceSynchronize());

	TimingClearAndStart(counter);
	switch (memAccess)
	{
	case maAlignedRead:
	case maPinnedAlignedRead:
		cuda_alignedRead<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maAlignedWrite:
	case maPinnedAlignedWrite:
		cuda_alignedWrite<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maNotAlignedRead:
	case maPinnedNotAlignedRead:
		cuda_notAlignedRead<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	case maNotAlignedWrite:
	case maPinnedNotAlignedWrite:
		cuda_notAlignedWrite<T>(dptr, count, repeatCount, blockCount, threadsPerBlock);
		break;
	}
	CUDA_SAFECALL(cudaDeviceSynchronize());
	TimingFinish(counter);

	return TimingSeconds(counter) / repeatCount;
}

double GPUBenchmarkCUDA::reductionSum_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<int>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCUDA::reductionSum_int64(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<__int64>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCUDA::reductionSum_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<float>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCUDA::reductionSum_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<double>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

template<typename T>
double GPUBenchmarkCUDA::reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int)
{
	TimingCounter counter;
	deviceMem<T> dData;
	deviceMem<T> dSum;
	deviceMem<T> dTemp;
	hostMem<T> hData;
	T sum;
	dData.allocate(count);
	dSum.allocate(1);
	dTemp.allocate(blockCount);
	hData.allocate(count);
	for (int i = 0; i < count; i++)
		hData[i] = (T)i;
	hData.copyTo(dData);
	CUDA_SAFECALL(cudaDeviceSynchronize());

	TimingClearAndStart(counter);
	cuda_reductionSum<T>(dData.dptr(), dSum.dptr(), dTemp.dptr(), count, repeatCount, blockCount, threadsPerBlock);
	CUDA_SAFECALL(cudaDeviceSynchronize());
	TimingFinish(counter);

	dSum.copyTo(&sum);
	double correctSum = ((double)(count - 1) / 2.0 * count);
	if (fabs(1.0 - double(sum) / correctSum) > 1e-3)
		printf("Reduction FAILED: Sum: %f, CorrectSum: %f\n", sum, correctSum);
	//assert(fabs(1.0 - double(sum) / ((double)(count - 1) / 2.0 * count)) < 1e-3);

	return TimingSeconds(counter) / repeatCount;
}

// template instantiation
template double GPUBenchmarkCUDA::deviceMemAccess<int>(int, int, int, int, int);
template double GPUBenchmarkCUDA::deviceMemAccess<float>(int, int, int, int, int);
template double GPUBenchmarkCUDA::deviceMemAccess<double>(int, int, int, int, int);

template double GPUBenchmarkCUDA::reductionSum<int>(int, int, int, int, int);
template double GPUBenchmarkCUDA::reductionSum<__int64>(int, int, int, int, int);
template double GPUBenchmarkCUDA::reductionSum<float>(int, int, int, int, int);
template double GPUBenchmarkCUDA::reductionSum<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelAdd<int>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd<__int64>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelAdd_indep<int>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd_indep<__int64>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd_indep<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelAdd_indep<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelMad<int>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMad<__int64>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMad<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMad<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelMadSF<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMadSF<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelMul<int>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMul<__int64>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMul<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelMul<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelDiv<int>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelDiv<__int64>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelDiv<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelDiv<double>(int, int, int, int, int);

template void GPUBenchmarkCUDA::kernelSin<float>(int, int, int, int, int);
template void GPUBenchmarkCUDA::kernelSin<double>(int, int, int, int, int);

#endif