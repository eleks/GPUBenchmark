// the following defines are added at build time
// #define T ...

__kernel void kernel_doTinyTask(int a, int b, __global float *result)
{
	int sum = a + b;

	int threadID = get_local_id(0);
	if (threadID >= 1024)
		*result = sum;
}

__kernel void kernel_doAdd_vector_float(int count, __global float *result)
{
	int bulkCount = count >> 6;
	int threadID = get_local_id(0);
	for (int i = 0; i < bulkCount; i++)
	{
		float4 value;
		value.x = i;
		value.y = value.x + 1.0f;
		value.z = value.x + 2.0f;
		value.w = value.x + 3.0f;

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

		if (threadID > 1024) // to avoid removal by optimization
			*result = value.x + value.y + value.z + value.w;
	}
}

__kernel void kernel_doAdd(int count, __global T *result)
{
	int bulkCount = count >> 5;
	int threadID = get_local_id(0);
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value;
	}
}

__kernel void kernel_doAdd_indep(int count, __global T *result)
{
	int bulkCount = count >> 6;
	int threadID = get_local_id(0);
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value1 + value2 + value3 + value4;
	}
}

__kernel void kernel_doMad(int count, __global T *result)
{
	int bulkCount = count >> 6;
	int threadID = get_local_id(0);
	for (int i = 0; i < bulkCount; i++)
	{
		T value1 = i, value2 = (T)1.0 + i, value3 = (T)2.0 + i, value4 = (T)3.0 + i;
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value1 + value2 + value3 + value4;
	}
}

__kernel void kernel_doMadSF(int count, __global T *result)
{
	int bulkCount = count >> 6;
	int threadID = get_local_id(0);
	for (int i = 0; i < bulkCount; i++)
	{
		T value1 = i, value2 = (T)1.0 + i, value3 = (T)2.0 + i, value4 = (T)3.0 + i;
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value1 + value2 + value3 + value4;
	}
}

__kernel void kernel_doMul(int count, __global T *result)
{
	int bulkCount = count >> 5;
	int threadID = get_local_id(0);
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value;
	}
}

__kernel void kernel_doDiv(int count, __global T *result)
{
	int bulkCount = count >> 5;
	int threadID = get_local_id(0);
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
		if (threadID > 1024) // to avoid removal by optimization
			*result = value;
	}
}

#ifdef FLOAT
__kernel void kernel_doSin(int count, __global T *result)
{
	int bulkCount = count >> 5;
	int threadID = get_local_id(0);
	for (int i = 0; i < bulkCount; i++)
	{
		T value = (T)1.0 + i;
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		value = native_sin(value);
		if (threadID > 1024) // to avoid removal by optimization
			*result = value;
	}
}
#endif

__kernel void kernel_reductionSum(__global T *data, __global T *sum, int count, 
	int repeatCount, __local T *ssum)
{
	for (int i = 0; i < repeatCount; i++)
	{
		size_t threadID = get_local_id(0);
		size_t localSize = get_local_size(0);
		size_t numGroups = get_num_groups(0);
		size_t groupID = get_group_id(0);
		size_t countPerGroup = (count + numGroups - 1) / numGroups;

		ssum[threadID] = 0;

		__global T *pBase = data + groupID * countPerGroup;
		__global T *pValue = pBase + threadID;
		__global T *pValueMax = pBase + countPerGroup;
		if (pValueMax > data + count)
			pValueMax = data + count;
		__global T *pResult = sum + groupID;

		while (pValue < pValueMax)
		{
			ssum[threadID] += *pValue;
			pValue += localSize;
		}
	    barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = localSize >> 1; i > 0; i >>= 1) 
		{
			if (threadID < i) 
				ssum[threadID] += ssum[threadID + i];
		    barrier(CLK_LOCAL_MEM_FENCE);
		}
	 	if (threadID == 0)
			*pResult = ssum[threadID];
	    barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void kernel_alignedRead(__global T *data, int count, int repeatCount)
{
	size_t threadID = get_local_id(0);
	size_t numGroups = get_num_groups(0);
	size_t countPerGroup = (count + numGroups - 1) / numGroups;
	size_t groupID = get_group_id(0);
	size_t localSize = get_local_size(0);
	__global T *pmax = data + groupID * countPerGroup + countPerGroup;
	size_t inc = localSize;
	for (int i = 0; i < repeatCount; i++)
	{
		__global T *p = data + groupID * countPerGroup + threadID;
		T sum = 0;
		while (p < pmax)
		{
			sum += *p;
			p += inc;
		}
		if (threadID > 1024) // to avoid removal by optimization
			*p = sum;
	}
}

__kernel void kernel_notAlignedRead(__global T *data, int count, int repeatCount)
{
	size_t threadID = get_local_id(0);
	size_t numGroups = get_num_groups(0);
	size_t countPerGroup = (count + numGroups - 1) / numGroups;
	size_t groupID = get_group_id(0);
	size_t localSize = get_local_size(0);
	size_t countPerThread = (countPerGroup + localSize - 1) / localSize;
	__global T *pmax = data + groupID * countPerGroup + threadID * countPerThread + countPerThread;
	size_t inc = 1;
	for (int i = 0; i < repeatCount; i++)
	{
		__global T *p = data + groupID * countPerGroup + threadID * countPerThread;
		T sum = 0;
		while (p < pmax)
		{
			sum += *p;
			p += inc;
		}
		if (threadID > 1024) // to avoid removal by optimization
			*p = sum;
	}
}

__kernel void kernel_alignedWrite(__global T *data, int count, int repeatCount)
{
	size_t threadID = get_local_id(0);
	size_t numGroups = get_num_groups(0);
	size_t countPerGroup = (count + numGroups - 1) / numGroups;
	size_t groupID = get_group_id(0);
	size_t localSize = get_local_size(0);
	__global T *pmax = data + groupID * countPerGroup + countPerGroup;
	size_t inc = localSize;
	for (int i = 0; i < repeatCount; i++)
	{
		__global T *p = data + groupID * countPerGroup + threadID;
		while (p < pmax)
		{
			*p = 0;
			p += inc;
		}
	}
}

__kernel void kernel_notAlignedWrite(__global T *data, int count, int repeatCount)
{
	size_t threadID = get_local_id(0);
	size_t numGroups = get_num_groups(0);
	size_t countPerGroup = (count + numGroups - 1) / numGroups;
	size_t groupID = get_group_id(0);
	size_t localSize = get_local_size(0);
	size_t countPerThread = (countPerGroup + localSize - 1) / localSize;
	__global T *pmax = data + groupID * countPerGroup + threadID * countPerThread + countPerThread;
	size_t inc = 1;
	for (int i = 0; i < repeatCount; i++)
	{
		__global T *p = data + groupID * countPerGroup + threadID * countPerThread;
		while (p < pmax)
		{
			*p = 0;
			p += inc;
		}
	}
}