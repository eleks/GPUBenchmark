#pragma once

#include <cstdlib>
#include <cstdio>

class GPUBenchmark;

typedef void (GPUBenchmark::*benchmarkOuterFunc)(int, int, int, int, int);
typedef double (GPUBenchmark::*benchmarkInnerFunc)(int, int, int, int, int);

enum DeviceMemAccessEnum : int {
	maAlignedRead, 
	maAlignedWrite,
	maNotAlignedRead, 
	maNotAlignedWrite,
	maPinnedAlignedRead, 
	maPinnedAlignedWrite,
	maPinnedNotAlignedRead, 
	maPinnedNotAlignedWrite
};

class GPUBenchmark
{
	DISALLOW_COPY_AND_ASSIGN(GPUBenchmark);

	double execOuterTiming(benchmarkOuterFunc func, int p1 = 0, int p2 = 0, int p3 = 0, int p4 = 0, int p5 = 0);
	double execInnerTiming(benchmarkInnerFunc func, int p1 = 0, int p2 = 0, int p3 = 0, int p4 = 0, int p5 = 0);
protected:
	__int64 deviceClockRate;

	virtual void deviceMemAllocRelease(int size, int repeatCount, int, int, int) = 0;
	virtual void mappedMemAllocRelease(int size, int repeatCount, int, int, int) = 0;
	virtual void hostMemRegisterUnregister(int size, int repeatCount, int, int, int) = 0;
	virtual void hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int) = 0;

	virtual double memTransfer(int size, int mode, int direction, int, int) = 0;

	virtual double deviceMemAccess_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) = 0;
	virtual double deviceMemAccess_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) = 0;
	virtual double deviceMemAccess_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) = 0;

	virtual double reductionSum_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int) = 0;
	virtual double reductionSum_int64(int count, int repeatCount, int blockCount, int threadsPerBlock, int) = 0;
	virtual double reductionSum_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int) = 0;
	virtual double reductionSum_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int) = 0;

	virtual double kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int) = 0;
	virtual double kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int) = 0;

	virtual void kernelAdd_int(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_int64(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelAdd_indep_int(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_indep_int64(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_indep_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelAdd_indep_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelMad_int(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMad_int64(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMad_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMad_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelMadSF_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMadSF_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelMul_int(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMul_int64(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMul_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelMul_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelDiv_int(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelDiv_int64(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelDiv_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelDiv_double(int count, int blockCount, int threadCount, int, int) = 0;

	virtual void kernelSin_float(int count, int blockCount, int threadCount, int, int) = 0;
	virtual void kernelSin_double(int count, int blockCount, int threadCount, int, int) = 0;
#ifdef CUDA50
	virtual double kernelDynamicExecuteTinyTask(int blockCount, int threadCount, int, int, int) = 0;
	virtual double kernelDynamicScheduleTinyTask(int blockCount, int threadCount, int, int, int) = 0;
#endif

	virtual void deviceInitialize() = 0;
	virtual void devicePropertiesShow() = 0;

	virtual bool isDynamicParallelismSupported() = 0;
	virtual bool isWriteCombinedMemorySupported() = 0;
	virtual bool isPageLockedMemorySupported() = 0;
	virtual bool isDoublePrecisionTrigonometrySupported() = 0;
	virtual bool isInt64Supported() = 0;
	virtual bool isPinnedMemorySupported() = 0;
public:
	GPUBenchmark() {};

	void run();
};
