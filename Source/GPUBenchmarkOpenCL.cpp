#ifdef OpenCL

#include <CL/cl.h>
#include "GPUBenchmarkOpenCL.h"

std::string GPUBenchmarkCL::sourceLoad(std::string &flp)
{
	AutoFile source(flp.c_str(), "rb");
 	source.seek(0, SEEK_END); 
    int length = source.tell();
    source.seek(0, SEEK_SET); 
	std::string result;
	result.resize(length + 1);
	source.read(&result[0], length);
	result[length] = '\0';
	return result;
}

void GPUBenchmarkCL::deviceInitialize()
{
	_context.initialize(CL_DEVICE_TYPE_GPU);
	_commandQueue.initialize(_context);
	std::string sourceFlp = _sourceDir + std::string("GPUBenchmark.cl");
	printf("Building OpenCL kernels from source (%s) ...\n", sourceFlp.c_str());
	std::string source = sourceLoad(sourceFlp);

	_programInt.initialize(_context, std::string("#define INT\n #define T int\n") + source);
	//_programInt64.initialize(_context, std::string("#define INT64\n #pragma OPENCL EXTENSION cl_khr_int64: enable\n #pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n #define T long long\n") + source);
	_programFloat.initialize(_context, std::string("#define FLOAT\n #define T float\n") + source);
	_programDouble.initialize(_context, std::string("#define DOUBLE\n #pragma OPENCL EXTENSION cl_khr_fp64: enable\n #define T double\n") + source);

	_programInt.build("-cl-std=CL1.1 -cl-fast-relaxed-math");
	//_programInt64.build("-cl-std=CL1.1 -cl-fast-relaxed-math");
	_programFloat.build("-cl-std=CL1.1 -cl-fast-relaxed-math");
	_programDouble.build("-cl-std=CL1.1 -cl-fast-relaxed-math");

	_kernel_doTinyTask.initialize(_programInt, "kernel_doTinyTask");

	_kernel_doAdd_int.initialize(_programInt, "kernel_doAdd");
	//_kernel_doAdd_int64.initialize(_programInt64, "kernel_doAdd");
	_kernel_doAdd_float.initialize(_programFloat, "kernel_doAdd");
	_kernel_doAdd_double.initialize(_programDouble, "kernel_doAdd");

	_kernel_doAdd_indep_int.initialize(_programInt, "kernel_doAdd_indep");
	//_kernel_doAdd_indep_int64.initialize(_programInt64, "kernel_doAdd_indep");
	_kernel_doAdd_indep_float.initialize(_programFloat, "kernel_doAdd_indep");
	_kernel_doAdd_indep_double.initialize(_programDouble, "kernel_doAdd_indep");

	_kernel_doMad_int.initialize(_programInt, "kernel_doMad");
	//_kernel_doMad_int64.initialize(_programInt64, "kernel_doMad");
	_kernel_doMad_float.initialize(_programFloat, "kernel_doMad");
	_kernel_doMad_double.initialize(_programDouble, "kernel_doMad");

	_kernel_doMadSF_float.initialize(_programFloat, "kernel_doMadSF");
	_kernel_doMadSF_double.initialize(_programDouble, "kernel_doMadSF");

	_kernel_doMul_int.initialize(_programInt, "kernel_doMul");
	//_kernel_doMul_int64.initialize(_programInt64, "kernel_doMul");
	_kernel_doMul_float.initialize(_programFloat, "kernel_doMul");
	_kernel_doMul_double.initialize(_programDouble, "kernel_doMul");

	_kernel_doDiv_int.initialize(_programInt, "kernel_doDiv");
	//_kernel_doDiv_int64.initialize(_programInt64, "kernel_doDiv");
	_kernel_doDiv_float.initialize(_programFloat, "kernel_doDiv");
	_kernel_doDiv_double.initialize(_programDouble, "kernel_doDiv");

	_kernel_doSin_float.initialize(_programFloat, "kernel_doSin");
	//_kernel_doSin_double.initialize(_programDouble, "kernel_doSin");

	_kernel_reductionSum_int.initialize(_programInt, "kernel_reductionSum");
	//_kernel_reductionSum_int64.initialize(_programInt64, "kernel_reductionSum");
	_kernel_reductionSum_float.initialize(_programFloat, "kernel_reductionSum");
	_kernel_reductionSum_double.initialize(_programDouble, "kernel_reductionSum");

	_kernel_alignedRead_int.initialize(_programInt, "kernel_alignedRead");
	_kernel_alignedRead_float.initialize(_programFloat, "kernel_alignedRead");
	_kernel_alignedRead_double.initialize(_programDouble, "kernel_alignedRead");

	_kernel_alignedWrite_int.initialize(_programInt, "kernel_alignedWrite");
	_kernel_alignedWrite_float.initialize(_programFloat, "kernel_alignedWrite");
	_kernel_alignedWrite_double.initialize(_programDouble, "kernel_alignedWrite");

	_kernel_notAlignedRead_int.initialize(_programInt, "kernel_notAlignedRead");
	_kernel_notAlignedRead_float.initialize(_programFloat, "kernel_notAlignedRead");
	_kernel_notAlignedRead_double.initialize(_programDouble, "kernel_notAlignedRead");

	_kernel_notAlignedWrite_int.initialize(_programInt, "kernel_notAlignedWrite");
	_kernel_notAlignedWrite_float.initialize(_programFloat, "kernel_notAlignedWrite");
	_kernel_notAlignedWrite_double.initialize(_programDouble, "kernel_notAlignedWrite");
	printf("Built successfully\n");

	//test(); //!!
}

void GPUBenchmarkCL::devicePropertiesShow()
{
	//!!
	printf("\n");
}

void GPUBenchmarkCL::deviceMemAllocRelease(int size, int repeatCount, int, int, int)
{
	CLDeviceMem<byte> dmem;
	for (int i = 0; i < repeatCount; i++)
	{
		dmem.allocate(_context, _commandQueue, size, CL_MEM_READ_WRITE);
		//!! buffer object is allocated on demand, so queuing tiny write operation
		CL_SAFECALL(clEnqueueWriteBuffer(_commandQueue.commandQueue(), 
			dmem.mem(), CL_TRUE, 0, sizeof(i), &i, 0, NULL, NULL));
		dmem.release();
		_commandQueue.finish();
	}
}

void GPUBenchmarkCL::mappedMemAllocRelease(int size, int repeatCount, int, int, int)
{
	CLMappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.allocate(_context, _commandQueue, size);
		mmem.map();
		mmem.release();
		_commandQueue.finish();
	}
}

void GPUBenchmarkCL::hostMemWriteCombinedAllocRelease(int size, int repeatCount, int, int, int)
{
	NOT_SUPPORTED("GPUBenchmarkCL::hostMemWriteCombinedAllocRelease");
}

void GPUBenchmarkCL::hostMemRegisterUnregister(int size, int repeatCount, int, int, int)
{
	CLHostMem<byte> hmem(size);
	CLMappedMem<byte> mmem;
	for (int i = 0; i < repeatCount; i++)
	{
		mmem.registerHost(_context, hmem.hptr(), size);
		mmem.release();
	}
}

double GPUBenchmarkCL::memTransfer(int size, int mode, int direction, int, int)
{
	const int ITERATIONS = 10;
	CLDeviceMem<byte> dmem;
	CLDeviceMem<byte> dmem2;
	CLHostMem<byte> hmem;
	CLMappedMem<byte> mmem;
	if (direction == 2)
		dmem2.allocate(_context, _commandQueue, size);
	switch (mode)
	{
	case 0:
		hmem.allocate(size);
		dmem.allocate(_context, _commandQueue, size);
		hmem.copyTo(dmem); // device memory initialization
		break;
	case 1:
		hmem.allocate(size);
		mmem.registerHost(_context, hmem.hptr(), size);
		dmem.allocate(_context, _commandQueue, size);
		//NOT_SUPPORTED("GPUBenchmarkCL::memTransfer: Page locked memory");
		break;
	case 2:
		NOT_SUPPORTED("GPUBenchmarkCL::memTransfer: Write combined memory");
		break;
	default:
		assert(false);
	}
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
			dmem.copyToAsync(dmem2);
			dmem.waitForCompletion();
			break;
		default:
			assert(false);
		}
	_commandQueue.finish();
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmarkCL::kernelExecuteTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
	{
		_kernel_doTinyTask.argSetInt(0, blockCount);
		_kernel_doTinyTask.argSetInt(1, threadCount);
		_kernel_doTinyTask.argSetPtr(2, NULL);
		_kernel_doTinyTask.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
		_kernel_doTinyTask.waitForCompletion();
	}
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

double GPUBenchmarkCL::kernelScheduleTinyTask(int blockCount, int threadCount, int, int, int)
{
	const int ITERATIONS = 1; //1000;
	TimingCounter counter;
	TimingClearAndStart(counter);
	for(int i = 0; i < ITERATIONS; i++)
	{
		_kernel_doTinyTask.argSetInt(0, blockCount);
		_kernel_doTinyTask.argSetInt(1, threadCount);
		_kernel_doTinyTask.argSetPtr(2, NULL);
		_kernel_doTinyTask.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	}
	TimingFinish(counter);

	return TimingSeconds(counter) / ITERATIONS;
}

void GPUBenchmarkCL::kernelAdd_int(int count, int blockCount, int threadCount, int, int) 
{
	return kernelAdd<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_int64(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelAdd(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_doAdd_int
		: typeid(T) == typeid(__int64) ? _kernel_doAdd_int64 
		: typeid(T) == typeid(float) ? _kernel_doAdd_float 
		: typeid(T) == typeid(double) ? _kernel_doAdd_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelAdd_indep_int(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd_indep<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_indep_int64(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd_indep<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_indep_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd_indep<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelAdd_indep_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelAdd_indep<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelAdd_indep(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_doAdd_indep_int
		: typeid(T) == typeid(__int64) ? _kernel_doAdd_indep_int64 
		: typeid(T) == typeid(float) ? _kernel_doAdd_indep_float 
		: typeid(T) == typeid(double) ? _kernel_doAdd_indep_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelMad_int(int count, int blockCount, int threadCount, int, int)
{
	return kernelMad<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMad_int64(int count, int blockCount, int threadCount, int, int)
{
	return kernelMad<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMad_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelMad<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMad_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelMad<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelMad(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_doMad_int
		: typeid(T) == typeid(__int64) ? _kernel_doMad_int64 
		: typeid(T) == typeid(float) ? _kernel_doMad_float 
		: typeid(T) == typeid(double) ? _kernel_doMad_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelMadSF_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelMadSF<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMadSF_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelMadSF<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelMadSF(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(float) ? _kernel_doMadSF_float 
		: typeid(T) == typeid(double) ? _kernel_doMadSF_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelMul_int(int count, int blockCount, int threadCount, int, int)
{
	return kernelMul<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMul_int64(int count, int blockCount, int threadCount, int, int)
{
	return kernelMul<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMul_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelMul<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelMul_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelMul<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelMul(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_doMul_int
		: typeid(T) == typeid(__int64) ? _kernel_doMul_int64 
		: typeid(T) == typeid(float) ? _kernel_doMul_float 
		: typeid(T) == typeid(double) ? _kernel_doMul_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelDiv_int(int count, int blockCount, int threadCount, int, int)
{
	return kernelDiv<int>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelDiv_int64(int count, int blockCount, int threadCount, int, int)
{
	return kernelDiv<__int64>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelDiv_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelDiv<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelDiv_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelDiv<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelDiv(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_doDiv_int
		: typeid(T) == typeid(__int64) ? _kernel_doDiv_int64 
		: typeid(T) == typeid(float) ? _kernel_doDiv_float 
		: typeid(T) == typeid(double) ? _kernel_doDiv_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

void GPUBenchmarkCL::kernelSin_float(int count, int blockCount, int threadCount, int, int)
{
	return kernelSin<float>(count, blockCount, threadCount, 0, 0);
}

void GPUBenchmarkCL::kernelSin_double(int count, int blockCount, int threadCount, int, int)
{
	return kernelSin<double>(count, blockCount, threadCount, 0, 0);
}

template<typename T>
void GPUBenchmarkCL::kernelSin(int count, int blockCount, int threadCount, int, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(float) ? _kernel_doSin_float 
		: typeid(T) == typeid(double) ? _kernel_doSin_double
		: (CLKernelBase&)_kernelNA;

	kernel.argSetInt(0, count);
	kernel.argSetPtr(1, NULL);
	kernel.enqueue1D(_commandQueue, blockCount * threadCount, threadCount);
	kernel.waitForCompletion();
}

double GPUBenchmarkCL::deviceMemAccess_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) 
{
	return deviceMemAccess<int>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

double GPUBenchmarkCL::deviceMemAccess_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) 
{
	return deviceMemAccess<float>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

double GPUBenchmarkCL::deviceMemAccess_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess) 
{
	return deviceMemAccess<double>(count, repeatCount, blockCount, threadsPerBlock, memAccess);
}

template<typename T> 
double GPUBenchmarkCL::deviceMemAccess(int count, int repeatCount, int blockCount, int threadsPerBlock, int memAccess)
{
	TimingCounter counter;
	CLDeviceMem<T> dData;
	CLMappedMem<T> mData;
	cl_mem mem;
	switch (memAccess)
	{
	case maPinnedAlignedRead:
	case maPinnedAlignedWrite:
	case maPinnedNotAlignedRead:
	case maPinnedNotAlignedWrite:
		mData.allocate(_context, _commandQueue, count);
		mData.map();
		memset(mData.hptr(), 0, count * sizeof(T));
		mem = mData.mem();
		break;
	default:
		dData.allocate(_context, _commandQueue, count);
		//dData.clear();
		mem = dData.mem();
	}
	_commandQueue.finish();

	CLKernelBase *kernel = &_kernelNA;

	TimingClearAndStart(counter);
	switch (memAccess)
	{
	case maAlignedRead:
	case maPinnedAlignedRead:
		kernel = typeid(T) == typeid(int) ? &_kernel_alignedRead_int
			: typeid(T) == typeid(float) ? &_kernel_alignedRead_float
			: typeid(T) == typeid(double) ? &_kernel_alignedRead_double
			: (CLKernelBase*)&_kernelNA;
		break;
	case maAlignedWrite:
	case maPinnedAlignedWrite:
		kernel = typeid(T) == typeid(int) ? &_kernel_alignedWrite_int
			: typeid(T) == typeid(float) ? &_kernel_alignedWrite_float
			: typeid(T) == typeid(double) ? &_kernel_alignedWrite_double
			: (CLKernelBase*)&_kernelNA;
		break;
	case maNotAlignedRead:
	case maPinnedNotAlignedRead:
		kernel = typeid(T) == typeid(int) ? &_kernel_notAlignedRead_int
			: typeid(T) == typeid(float) ? &_kernel_notAlignedRead_float
			: typeid(T) == typeid(double) ? &_kernel_notAlignedRead_double
			: (CLKernelBase*)&_kernelNA;
		break;
	case maNotAlignedWrite:
	case maPinnedNotAlignedWrite:
		kernel = typeid(T) == typeid(int) ? &_kernel_notAlignedWrite_int
			: typeid(T) == typeid(float) ? &_kernel_notAlignedWrite_float
			: typeid(T) == typeid(double) ? &_kernel_notAlignedWrite_double
			: (CLKernelBase*)&_kernelNA;
		break;
	}
	kernel->argSetMem(0, mem);
	kernel->argSetInt(1, count);
	kernel->argSetInt(2, repeatCount);
	kernel->enqueue1D(_commandQueue, blockCount * threadsPerBlock, threadsPerBlock);
	kernel->waitForCompletion();
	TimingFinish(counter);

	return TimingSeconds(counter) / repeatCount;
}

double GPUBenchmarkCL::reductionSum_int(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<int>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCL::reductionSum_int64(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<__int64>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCL::reductionSum_float(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<float>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

double GPUBenchmarkCL::reductionSum_double(int count, int repeatCount, int blockCount, int threadsPerBlock, int) 
{
	return reductionSum<double>(count, repeatCount, blockCount, threadsPerBlock, 0);
}

template<typename T>
double GPUBenchmarkCL::reductionSum(int count, int repeatCount, int blockCount, int threadsPerBlock, int)
{
	CLKernelBase &kernel = typeid(T) == typeid(int) ? _kernel_reductionSum_int
		: typeid(T) == typeid(__int64) ? _kernel_reductionSum_int64 
		: typeid(T) == typeid(float) ? _kernel_reductionSum_float 
		: typeid(T) == typeid(double) ? _kernel_reductionSum_double
		: (CLKernelBase&)_kernelNA;

	TimingCounter counter;
	CLDeviceMem<T> dData;
	CLDeviceMem<T> dSum;
	CLDeviceMem<T> dTemp;
	CLHostMem<T> hData;
	T sum;
	dData.allocate(_context, _commandQueue, count);
	dSum.allocate(_context, _commandQueue, 1);
	dTemp.allocate(_context, _commandQueue, blockCount);
	hData.allocate(count);
	for (int i = 0; i < count; i++)
		hData[i] = (T)i;
	hData.copyTo(dData);
	_commandQueue.finish();

	TimingClearAndStart(counter);

	kernel.argSetMem(0, dData.mem());
	kernel.argSetMem(1, dTemp.mem());
	kernel.argSetInt(2, count);
	kernel.argSetInt(3, repeatCount);
	kernel.argSetSharedMem(4, sizeof(T) * threadsPerBlock);
	kernel.enqueue1D(_commandQueue, blockCount * threadsPerBlock, threadsPerBlock);
	//kernel.waitForCompletion();

	kernel.argSetMem(0, dTemp.mem());
	kernel.argSetMem(1, dSum.mem());
	kernel.argSetInt(2, blockCount);
	kernel.argSetInt(3, repeatCount);
	kernel.argSetSharedMem(4, sizeof(T) * threadsPerBlock);
	kernel.enqueue1D(_commandQueue, threadsPerBlock, threadsPerBlock);
	kernel.waitForCompletion();

	TimingFinish(counter);

	dSum.copyTo(&sum);
	double correctSum = ((double)(count - 1) / 2.0 * count);
	if (fabs(1.0 - double(sum) / correctSum) > 1e-3)
		printf("Reduction FAILED: Sum: %f, CorrectSum: %f\n", sum, correctSum);

	return TimingSeconds(counter) / repeatCount;
}

void GPUBenchmarkCL::test()
{
	return;
	{
		CLHostMem<int> hmem;
		hmem.allocate(100 * 1024 * 1024);

		CLDeviceMem<int> m1, m2, m3, m4, m5, m6;
		m1.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m2.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m3.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m4.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m5.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m6.allocate(_context, _commandQueue, 100 * 1024 * 1024);

		hmem.copyTo(m1);
		hmem.copyTo(m2);
		hmem.copyTo(m3);
		hmem.copyTo(m4);
		hmem.copyTo(m5);
		hmem.copyTo(m6);
	}

	{
		CLHostMem<byte> hmem;
		CLMappedMem<int> m1, m2, m3, m4, m5, m6;
		m1.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		//m1.map();
		m2.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		//m2.map();
		m3.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		//m3.map();
		m4.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		//m4.map();
		m5.allocate(_context, _commandQueue, 100 * 1024 * 1024);
		m6.allocate(_context, _commandQueue, 100 * 1024 * 1024);
	}
}

// template instantiation
template double GPUBenchmarkCL::deviceMemAccess<int>(int, int, int, int, int);
template double GPUBenchmarkCL::deviceMemAccess<float>(int, int, int, int, int);
template double GPUBenchmarkCL::deviceMemAccess<double>(int, int, int, int, int);

template double GPUBenchmarkCL::reductionSum<int>(int, int, int, int, int);
template double GPUBenchmarkCL::reductionSum<__int64>(int, int, int, int, int);
template double GPUBenchmarkCL::reductionSum<float>(int, int, int, int, int);
template double GPUBenchmarkCL::reductionSum<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelAdd<int>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd<__int64>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelAdd_indep<int>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd_indep<__int64>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd_indep<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelAdd_indep<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelMad<int>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMad<__int64>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMad<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMad<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelMadSF<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMadSF<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelMul<int>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMul<__int64>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMul<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelMul<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelDiv<int>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelDiv<__int64>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelDiv<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelDiv<double>(int, int, int, int, int);

template void GPUBenchmarkCL::kernelSin<float>(int, int, int, int, int);
template void GPUBenchmarkCL::kernelSin<double>(int, int, int, int, int);

#endif