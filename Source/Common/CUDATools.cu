#ifdef CUDA

#include <device_types.h>
#include "CUDATools.h"
#include "CUDADeviceTools.h"

template<typename T>
__global__ void kernel_arraySub(T *A, T b, int count)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < count; i += blockDim.x * gridDim.x
	)
		A[i] -= b;
}

template<typename T> 
void cuda_arraySub(T *A, T b, int count)
{
	int blockCount = (count + CUDA_TOOLS_THREADS_PER_BLOCK - 1) / CUDA_TOOLS_THREADS_PER_BLOCK;
	kernel_arraySub<T><<<min(blockCount, CUDA_TOOLS_MAX_BLOCKS), CUDA_TOOLS_THREADS_PER_BLOCK>>>(
		A, b, count);
}

template<typename T>
__global__ void kernel_arrayMax(T *R, T *A, T b, int count)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
		i < count; i += blockDim.x * gridDim.x
	)
		R[i] = max(A[i], b);
}

template<typename T> 
void cuda_arrayMax(T *R, T *A, T b, int count)
{
	int blockCount = (count + CUDA_TOOLS_THREADS_PER_BLOCK - 1) / CUDA_TOOLS_THREADS_PER_BLOCK;
	kernel_arrayMax<T><<<min(blockCount, CUDA_TOOLS_MAX_BLOCKS), CUDA_TOOLS_THREADS_PER_BLOCK>>>(
		R, A, b, count);
}

template<typename T>
__global__ void kernel_arraySum(T *R, T *A, int count)
{
	__shared__ T ssum[CUDA_TOOLS_THREADS_PER_BLOCK];

	unsigned int threadID = threadIdx.x;
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;

	ssum[threadID] = 0;

	T *pBase = A + blockIdx.x * countPerBlock;
	T *pValue = pBase + threadID;
	T *pValueMax = pBase + countPerBlock;
	if (pValueMax > A + count)
		pValueMax = A + count;
	T *pResult = R + blockIdx.x;

	while (pValue < pValueMax)
	{
		ssum[threadID] += *pValue;
		pValue += CUDA_TOOLS_THREADS_PER_BLOCK;
	}
	__syncthreads();

    for (int i = CUDA_TOOLS_THREADS_PER_BLOCK >> 1; i > 0; i >>= 1) 
    {
        if (threadID < i) 
			ssum[threadID] += ssum[threadID + i];
		__syncthreads();
    }

	if (threadID == 0)
		*pResult = ssum[threadID];
	__syncthreads();
}

template<typename T> 
void cuda_arraySum(T *R, T *A, int count)
{
	int blockCount = min((count + CUDA_TOOLS_THREADS_PER_BLOCK - 1) 
		/ CUDA_TOOLS_THREADS_PER_BLOCK, CUDA_TOOLS_MAX_BLOCKS);
	deviceMem<T> d_temp(blockCount);
	kernel_arraySum<T><<<min(blockCount, CUDA_TOOLS_MAX_BLOCKS), CUDA_TOOLS_THREADS_PER_BLOCK>>>(
		d_temp.dptr(), A, count);
	kernel_arraySum<T><<<1, CUDA_TOOLS_THREADS_PER_BLOCK>>>(
		R, d_temp.dptr(), blockCount);
}

template<typename T>
__global__ void kernel_arrayStd_step1(T *R, T *A, int count, T *sum)
{
	__shared__ T ssum[CUDA_TOOLS_THREADS_PER_BLOCK];

	unsigned int threadID = threadIdx.x;
	unsigned int countPerBlock = (count + gridDim.x - 1) / gridDim.x;

	T mean = *sum / (T)count;
	ssum[threadID] = 0;

	T *pBase = A + blockIdx.x * countPerBlock;
	T *pValue = pBase + threadID;
	T *pValueMax = pBase + countPerBlock;
	if (pValueMax > A + count)
		pValueMax = A + count;
	T *pResult = R + blockIdx.x;

	while (pValue < pValueMax)
	{
		T sub = *pValue - mean;
		ssum[threadID] += sub * sub;
		pValue += CUDA_TOOLS_THREADS_PER_BLOCK;
	}
	__syncthreads();

    for (int i = CUDA_TOOLS_THREADS_PER_BLOCK >> 1; i > 0; i >>= 1) 
    {
        if (threadID < i)
			ssum[threadID] += ssum[threadID + i];
		__syncthreads();
    }

	if (threadID == 0)
		*pResult = ssum[threadID];
	__syncthreads();
}

template<typename T>
__global__ void kernel_arrayStd_step2(T *S, int count)
{
	*S = sqrt((double)(*S / (T)(count - 1)));
}

template<typename T> 
void cuda_arrayStd(T *R, T *A, int count)
{
	int blockCount = min((count + CUDA_TOOLS_THREADS_PER_BLOCK - 1) 
		/ CUDA_TOOLS_THREADS_PER_BLOCK, CUDA_TOOLS_MAX_BLOCKS);
	deviceMem<T> d_sum(1);
	cuda_arraySum(d_sum.dptr(), A, count);

	deviceMem<T> d_temp(blockCount);
	kernel_arrayStd_step1<T><<<blockCount, CUDA_TOOLS_THREADS_PER_BLOCK>>>(d_temp.dptr(), A, count, d_sum.dptr());
	kernel_arraySum<T><<<1, CUDA_TOOLS_THREADS_PER_BLOCK>>>(R, d_temp.dptr(), blockCount);
	kernel_arrayStd_step2<T><<<1, 1>>>(R, count);
}

// template instantiation
template void cuda_arraySub<int>(int *A, int b, int count);
template void cuda_arraySub<float>(float *A, float b, int count);
template void cuda_arraySub<double>(double *A, double b, int count);

template void cuda_arrayMax<int>(int *R, int *A, int b, int count);
template void cuda_arrayMax<float>(float *R, float *A, float b, int count);
template void cuda_arrayMax<double>(double *R, double *A, double b, int count);

template void cuda_arraySum<int>(int *R, int *A, int count);
template void cuda_arraySum<float>(float *R, float *A, int count);
template void cuda_arraySum<double>(double *R, double *A, int count);

template void cuda_arrayStd<int>(int *R, int *A, int count);
template void cuda_arrayStd<float>(float *R, float *A, int count);
template void cuda_arrayStd<double>(double *R, double *A, int count);

#endif