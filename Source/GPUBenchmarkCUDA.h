#ifdef CUDA

#pragma once

#include "GPUBenchmark.h"

class GPUBenchmarkCUDA : public GPUBenchmark
{
private:
	template<typename T> double deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess);
	template<typename T> double reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int);
	template<typename T> void kernelAdd(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelAdd_indep(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMad(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMadSF(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMul(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelDiv(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelSin(int count, int blockCount, int threadCount, int, int);
protected:
	void deviceMemAllocRelease(int size, int repeatCount, int, int, int) OVERRIDE;
	void mappedMemAllocRelease(int size, int repeatCount, int, int, int) OVERRIDE;
	void hostMemRegisterUnregister(int size, int repeatCount, int, int, int) OVERRIDE;
	void hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int) OVERRIDE;

	double memTransfer(int size, int mode, int direction, int, int) OVERRIDE;

	double deviceMemAccess_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) OVERRIDE;
	double deviceMemAccess_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) OVERRIDE;
	double deviceMemAccess_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) OVERRIDE;

	double reductionSum_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int) OVERRIDE;
	double reductionSum_int64(int count, int repeatCount, int blockCount, int threadsPerBlock, int) OVERRIDE;
	double reductionSum_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int) OVERRIDE;
	double reductionSum_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int) OVERRIDE;

	double kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int) OVERRIDE;
	double kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int) OVERRIDE;

	void kernelAdd_int(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_int64(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelAdd_indep_int(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_indep_int64(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_indep_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelAdd_indep_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelMad_int(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMad_int64(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMad_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMad_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelMadSF_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMadSF_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelMul_int(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMul_int64(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMul_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelMul_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelDiv_int(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelDiv_int64(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelDiv_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelDiv_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void kernelSin_float(int count, int blockCount, int threadCount, int, int) OVERRIDE;
	void kernelSin_double(int count, int blockCount, int threadCount, int, int) OVERRIDE;

	void deviceInitialize() OVERRIDE;
	void devicePropertiesShow() OVERRIDE;

	virtual bool isDynamicParallelismSupported() OVERRIDE {
		#ifdef CUDA50
			return true;
		#else
			return false;
		#endif
	}
	virtual bool isWriteCombinedMemorySupported() OVERRIDE {
		return true;
	}
	virtual bool isPageLockedMemorySupported() OVERRIDE {
		return true;
	}
	virtual bool isDoublePrecisionTrigonometrySupported() OVERRIDE {
		return true;
	}
	virtual bool isInt64Supported() OVERRIDE {
		return true;
	}
	virtual bool isPinnedMemorySupported() OVERRIDE {
		return true;
	}
public:
	GPUBenchmarkCUDA() {};
};

template<typename T> 
void cuda_reductionSum(T *data, T *sum, T *temp, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_alignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_notAlignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_alignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);
template<typename T> 
void cuda_notAlignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock);

template<typename T> 
void cuda_doAdd(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doAdd_indep(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doMad(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doMadSF(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doMul(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doDiv(int count, int blockCount, int threadCount);
template<typename T> 
void cuda_doSin(int count, int blockCount, int threadCount);

void cuda_doTinyTask(int blockCount, int threadCount);
#ifdef CUDA50
double cuda_doDynamicTinyTask(int blockCount, int threadCount, bool waitForCompletion);
#endif

#endif