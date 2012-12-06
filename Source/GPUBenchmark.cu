#ifdef CUDA

#include <device_types.h>
#include "Common/CUDATools.h"
#include "Common/CUDADeviceTools.h"

template<typename T, size_t threadsPerBlock>
__global__ void kernel_reductionSum(T *data, T *sum, int count, int repeatCount)
{
	__shared__ T ssum[threadsPerBlock];

	for (int i = 0; i < repeatCount; i++)
	{
		unsigned int threadID = threadIdx.x;
		unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;

		ssum[threadID] = 0;

		T *pBase = data + blockIdx.x * countPerBlock;
		T *pValue = pBase + threadID;
		T *pValueMax = pBase + countPerBlock;
		if (pValueMax > data + count)
			pValueMax = data + count;
		T *pResult = sum + blockIdx.x;

		while (pValue < pValueMax)
		{
			ssum[threadID] += *pValue;
			pValue += blockDim.x;
		}
		__syncthreads();

		for (int i = blockDim.x >> 1; i > 16; i >>= 1) 
		{
			if (threadID < i) 
				ssum[threadID] += ssum[threadID + i];
			__syncthreads();
		}

	#ifdef CUDA50_
		T value = ssum[threadID];
		if (sizeof(T) == sizeof(int))
		{
			value += __shfl_xor((T)value, 16, 32);
			value += __shfl_xor((T)value, 8, 32);
			value += __shfl_xor((T)value, 4, 32);
			value += __shfl_xor((T)value, 2, 32);
			value += __shfl_xor((T)value, 1, 32);
		} else
		if (sizeof(T) == sizeof(double))
		{
			//!!
		}
		if (threadID == 0)
			*pResult = value;
	#else
		#pragma unroll
		for (int i = 16; i > 0; i >>= 1) 
		{
			if (threadID < i) 
				ssum[threadID] += ssum[threadID + i];
			__syncthreads();
		}
		if (threadID == 0)
			*pResult = ssum[threadID];
	#endif
		__syncthreads();
	}
}

template<typename T> 
__global__ void kernel_alignedRead(T *data, int count, int repeatCount)
{
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;
	T *pmax = data + blockIdx.x * countPerBlock + countPerBlock;
	size_t inc = blockDim.x;
	for (int i = 0; i < repeatCount; i++)
	{
		T *p = data + blockIdx.x * countPerBlock + threadIdx.x;
		T sum = 0;
		while (p < pmax)
		{
			sum += *p;
			p += inc;
		}
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f\n", sum);
	}
}

template<typename T> 
__global__ void kernel_notAlignedRead(T *data, int count, int repeatCount)
{
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;
	unsigned int countPerThread = (countPerBlock + blockDim.x - 1) / blockDim.x;
	T *pmax = data + blockIdx.x * countPerBlock + threadIdx.x * countPerThread + countPerThread;
	size_t inc = 1;
	for (int i = 0; i < repeatCount; i++)
	{
		T *p = data + blockIdx.x * countPerBlock + threadIdx.x * countPerThread;
		T sum = 0;
		while (p < pmax)
		{
			sum += *p;
			p += inc;
		}
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f\n", sum);
	}
}

template<typename T> 
__global__ void kernel_alignedWrite(T *data, int count, int repeatCount)
{
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;
	T *pmax = data + blockIdx.x * countPerBlock + countPerBlock;
	size_t inc = blockDim.x;
	for (int i = 0; i < repeatCount; i++)
	{
		T *p = data + blockIdx.x * countPerBlock + threadIdx.x;
		while (p < pmax)
		{
			*p = 0;
			p += inc;
		}
	}
}

template<typename T> 
__global__ void kernel_notAlignedWrite(T *data, int count, int repeatCount)
{
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;
	unsigned int countPerThread = (countPerBlock + blockDim.x - 1) / blockDim.x;
	T *pmax = data + blockIdx.x * countPerBlock + threadIdx.x * countPerThread + countPerThread;
	size_t inc = 1;
	for (int i = 0; i < repeatCount; i++)
	{
		T *p = data + blockIdx.x * countPerBlock + threadIdx.x * countPerThread;
		while (p < pmax)
		{
			*p = 0;
			p += inc;
		}
	}
}

template<typename T> 
void cuda_alignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock)
{
	kernel_alignedRead<T><<<blockCount, threadsPerBlock>>>(data, count, repeatCount);
}

template<typename T> 
void cuda_notAlignedRead(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock)
{
	kernel_notAlignedRead<T><<<blockCount, threadsPerBlock>>>(data, count, repeatCount);
}

template<typename T> 
void cuda_alignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock)
{
	kernel_alignedWrite<T><<<blockCount, threadsPerBlock>>>(data, count, repeatCount);
}

template<typename T> 
void cuda_notAlignedWrite(T *data, int count, int repeatCount, int blockCount, int threadsPerBlock)
{
	kernel_notAlignedWrite<T><<<blockCount, threadsPerBlock>>>(data, count, repeatCount);
}

template<typename T> 
void cuda_reductionSum(T *data, T *sum, T *temp, int count, int repeatCount, int blockCount, int threadsPerBlock)
{
	switch (threadsPerBlock)
	{
	case 1:
	case 2:
	case 4:
	case 8:
	case 16:
	case 32:
		kernel_reductionSum<T, 32><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 32><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	case 64:
		kernel_reductionSum<T, 64><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 64><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	case 128:
		kernel_reductionSum<T, 128><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 128><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	case 256:
		kernel_reductionSum<T, 256><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 256><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	case 512:
		kernel_reductionSum<T, 512><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 512><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	case 1024:
		kernel_reductionSum<T, 1024><<<blockCount, threadsPerBlock>>>(data, temp, count, repeatCount);
		kernel_reductionSum<T, 1024><<<1, threadsPerBlock>>>(temp, sum, blockCount, repeatCount);
		break;
	}
}

__global__ void kernel_doTinyTask(int a, int b)
{
	int sum = a + b;
	if (threadIdx.x > 1024) // to avoid removal by optimization
		printf("%d", sum);
}

void cuda_doTinyTask(int blockCount, int threadCount)
{
	kernel_doTinyTask<<<blockCount, threadCount>>>(blockCount, threadCount);
}

template<typename T> 
__global__ void kernel_doAdd(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		T value = i;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		value = value + value;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f", value);
	}
}
/*
template<> 
__global__ void kernel_doAdd<float>(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		float4 value;
		value.x = i;
		value.y = value.x + 1.0f;
		value.z = value.x + 2.0f;
		value.w = value.x + 3.0f;

		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		value.x = value.x + value.x;
		value.y = value.y + value.y;
		value.z = value.z + value.z;
		value.w = value.w + value.w;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f,%f,%f,%f", value.x, value.y, value.z, value.w);
	}
}
*/
template<typename T> 
void cuda_doAdd(int count, int blockCount, int threadCount)
{
	kernel_doAdd<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doAdd_indep(int count)
{
	int bulkCount = count >> 6;
	for (int i = 0; i < bulkCount; i++)
	{
		T value1 = i, value2 = (T)1.0 + i, value3 = (T)2.0 + i, value4 = (T)3.0 + i;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		value1 = value1 + value1;
		value2 = value2 + value2;
		value3 = value3 + value3;
		value4 = value4 + value4;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f, %f, %f, %f", value1, value2, value3, value4);
	}
}

template<typename T> 
void cuda_doAdd_indep(int count, int blockCount, int threadCount)
{
	kernel_doAdd_indep<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doMad(int count)
{
	int bulkCount = count >> 6;
	for (int i = 0; i < bulkCount; i++)
	{
		T value1 = i, value2 = i + (T)1.0, value3 = i + (T)2.0, value4 = i + (T)3.0;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		value1 = value1 + value1 * value1;
		value2 = value2 + value2 * value2;
		value3 = value3 + value3 * value3;
		value4 = value4 + value4 * value4;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f,%f,%f,%f", value1, value2, value3, value4);
	}
}

template<typename T> 
void cuda_doMad(int count, int blockCount, int threadCount)
{
	kernel_doMad<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doMadSF(int count)
{
	int bulkCount = count >> 6;
	for (int i = 0; i < bulkCount; i++)
	{
		T value1 = i, value2 = i + (T)1.0, value3 = i + (T)2.0, value4 = i + (T)3.0;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		value1 = value1 + value1 * value1;
		value2 = sin(value2);
		value3 = sqrt(value3);
		value4 = value4 / value4;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("sum: %f,%f,%f,%f", value1, value2, value3, value4);
	}
}

template<typename T> 
void cuda_doMadSF(int count, int blockCount, int threadCount)
{
	kernel_doMadSF<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doMul(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		T value = (T)i;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		value = value * value;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("%f", value);
	}
}

template<typename T> 
void cuda_doMul(int count, int blockCount, int threadCount)
{
	kernel_doMul<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doDiv(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		T value = (T)i + (T)1.2345;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		value = value / value;
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("%f", value);
	}
}

template<typename T> 
void cuda_doDiv(int count, int blockCount, int threadCount)
{
	kernel_doDiv<T><<<blockCount, threadCount>>>(count);
}

template<typename T> 
__global__ void kernel_doSin(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		T value = (T)1.0 + i;
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		value = sin(value);
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("%f", value);
	}
}
/*
template<> 
__global__ void kernel_doSin<float>(int count)
{
	int bulkCount = count >> 5;
	for (int i = 0; i < bulkCount; i++)
	{
		float value = 1.0f + i;
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		value = sinf(value);
		if (threadIdx.x > 1024) // to avoid removal by optimization
			printf("%f", value);
	}
}
*/
template<typename T> 
void cuda_doSin(int count, int blockCount, int threadCount)
{
	kernel_doSin<T><<<blockCount, threadCount>>>(count);
}

#ifdef CUDA50
template<bool waitForCompletion>
__global__ void kernel_doDynamicTinyTask(int blockCount, int threadCount, 
	double *time)
{
	DTimingCounter counter;
	DTimingClearAndStart(counter);
	for (int i = 0; i < 1000; i++)
	{
		kernel_doTinyTask<<<blockCount, threadCount>>>(i, i);
		if (waitForCompletion)
			cudaDeviceSynchronize();
	}
	DTimingFinish(counter);
	*time = DTimingSeconds(counter) / 1000;
}

double cuda_doDynamicTinyTask(int blockCount, int threadCount, bool waitForCompletion)
{
	deviceMem<double> d_time(1);
	if (waitForCompletion)
		kernel_doDynamicTinyTask<true><<<1, 1>>>(blockCount, threadCount, d_time.dptr);
	else
		kernel_doDynamicTinyTask<false><<<1, 1>>>(blockCount, threadCount, d_time.dptr);
	cudaSafeCall(cudaThreadSynchronize());
	double result;
	d_time.copyTo(&result);
	return result;
}
#endif

// template instantiation
template void cuda_reductionSum<int>(int *, int *, int *, int, int, int, int);
template void cuda_reductionSum<__int64>(__int64 *, __int64 *, __int64 *, int, int, int, int);
template void cuda_reductionSum<float>(float *, float *, float *, int, int, int, int);
template void cuda_reductionSum<double>(double *, double *, double *, int, int, int, int);

template void cuda_alignedRead<int>(int *, int, int, int, int);
template void cuda_alignedRead<__int64>(__int64 *, int, int, int, int);
template void cuda_alignedRead<float>(float *, int, int, int, int);
template void cuda_alignedRead<double>(double *, int, int, int, int);

template void cuda_notAlignedRead<int>(int *, int, int, int, int);
template void cuda_notAlignedRead<__int64>(__int64 *, int, int, int, int);
template void cuda_notAlignedRead<float>(float *, int, int, int, int);
template void cuda_notAlignedRead<double>(double *, int, int, int, int);

template void cuda_alignedWrite<int>(int *, int, int, int, int);
template void cuda_alignedWrite<__int64>(__int64 *, int, int, int, int);
template void cuda_alignedWrite<float>(float *, int, int, int, int);
template void cuda_alignedWrite<double>(double *, int, int, int, int);

template void cuda_notAlignedWrite<int>(int *, int, int, int, int);
template void cuda_notAlignedWrite<__int64>(__int64 *, int, int, int, int);
template void cuda_notAlignedWrite<float>(float *, int, int, int, int);
template void cuda_notAlignedWrite<double>(double *, int, int, int, int);

template void cuda_doAdd<int>(int, int, int);
template void cuda_doAdd<__int64>(int, int, int);
template void cuda_doAdd<float>(int, int, int);
template void cuda_doAdd<double>(int, int, int);

template void cuda_doAdd_indep<int>(int, int, int);
template void cuda_doAdd_indep<__int64>(int, int, int);
template void cuda_doAdd_indep<float>(int, int, int);
template void cuda_doAdd_indep<double>(int, int, int);

template void cuda_doMad<int>(int, int, int);
template void cuda_doMad<__int64>(int, int, int);
template void cuda_doMad<float>(int, int, int);
template void cuda_doMad<double>(int, int, int);

template void cuda_doMadSF<float>(int, int, int);
template void cuda_doMadSF<double>(int, int, int);

template void cuda_doMul<int>(int, int, int);
template void cuda_doMul<__int64>(int, int, int);
template void cuda_doMul<float>(int, int, int);
template void cuda_doMul<double>(int, int, int);

template void cuda_doDiv<int>(int, int, int);
template void cuda_doDiv<__int64>(int, int, int);
template void cuda_doDiv<float>(int, int, int);
template void cuda_doDiv<double>(int, int, int);

template void cuda_doSin<float>(int, int, int);
template void cuda_doSin<double>(int, int, int);

#endif