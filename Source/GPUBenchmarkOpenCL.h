#ifdef OpenCL

#pragma once

#include "Common/OpenCLTools.h"
#include "GPUBenchmark.h"

class GPUBenchmarkCL : public GPUBenchmark
{
private:
	CLContext _context;
	CLCommandQueue _commandQueue;
	CLProgram _programInt;
	CLProgram _programInt64;
	CLProgram _programFloat;
	CLProgram _programDouble;

	CLKernelNA _kernelNA;
	CLKernel _kernel_doTinyTask;

	CLKernel _kernel_doAdd_int;
	CLKernel _kernel_doAdd_int64;
	CLKernel _kernel_doAdd_float;
	CLKernel _kernel_doAdd_double;

	CLKernel _kernel_doAdd_indep_int;
	CLKernel _kernel_doAdd_indep_int64;
	CLKernel _kernel_doAdd_indep_float;
	CLKernel _kernel_doAdd_indep_double;

	CLKernel _kernel_doMad_int;
	CLKernel _kernel_doMad_int64;
	CLKernel _kernel_doMad_float;
	CLKernel _kernel_doMad_double;

	CLKernel _kernel_doMadSF_float;
	CLKernel _kernel_doMadSF_double;

	CLKernel _kernel_doMul_int;
	CLKernel _kernel_doMul_int64;
	CLKernel _kernel_doMul_float;
	CLKernel _kernel_doMul_double;

	CLKernel _kernel_doDiv_int;
	CLKernel _kernel_doDiv_int64;
	CLKernel _kernel_doDiv_float;
	CLKernel _kernel_doDiv_double;

	CLKernel _kernel_doSin_float;
	CLKernel _kernel_doSin_double;

	CLKernel _kernel_reductionSum_int;
	CLKernel _kernel_reductionSum_int64;
	CLKernel _kernel_reductionSum_float;
	CLKernel _kernel_reductionSum_double;

	CLKernel _kernel_alignedRead_int;
	CLKernel _kernel_alignedRead_float;
	CLKernel _kernel_alignedRead_double;

	CLKernel _kernel_alignedWrite_int;
	CLKernel _kernel_alignedWrite_float;
	CLKernel _kernel_alignedWrite_double;

	CLKernel _kernel_notAlignedRead_int;
	CLKernel _kernel_notAlignedRead_float;
	CLKernel _kernel_notAlignedRead_double;

	CLKernel _kernel_notAlignedWrite_int;
	CLKernel _kernel_notAlignedWrite_float;
	CLKernel _kernel_notAlignedWrite_double;

	std::string _sourceDir;

	template<typename T> double deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess);
	template<typename T> double reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int);
	template<typename T> void kernelAdd(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelAdd_indep(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMad(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMadSF(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelMul(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelDiv(int count, int blockCount, int threadCount, int, int);
	template<typename T> void kernelSin(int count, int blockCount, int threadCount, int, int);

	std::string sourceLoad(std::string &flp);

	void test();
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
		return false;
	}
	virtual bool isWriteCombinedMemorySupported() OVERRIDE {
		#ifdef OpenCL12
			return true;
		#else
			return false;
		#endif
	}
	virtual bool isPageLockedMemorySupported() OVERRIDE {
		return true;
	}
	virtual bool isDoublePrecisionTrigonometrySupported() OVERRIDE {
		return false;
	}
	virtual bool isInt64Supported() OVERRIDE {
		return false; //!! OpenCL 1.1 has strange support of Int64 (long long), compiler fails in some cases
	}
	virtual bool isPinnedMemorySupported() OVERRIDE {
		return false; //!! OpenCL 1.1 caches mapped regions so behaviour is different (comparing to CUDA)
	}
public:
	GPUBenchmarkCL(std::string sourceDir) : _sourceDir(sourceDir) {};
};

#endif