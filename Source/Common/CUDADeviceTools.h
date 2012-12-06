#ifdef CUDA

#pragma once

//#define DEVICE_CLOCK_FREQUENCY 1215000000 //!! GTX 470
//#define DEVICE_CLOCK_FREQUENCY 1544000000 //!! GTX 580
#define DEVICE_CLOCK_FREQUENCY 1

#ifdef linux
typedef long long __int64;
#endif

#define Assert(c) if (!(c)) {printf("assert failed\n"); return;}

struct DTimingCounter {
	__int64 Counter;
	__int64 CounterTotal;

	__device__ DTimingCounter() : Counter(0), CounterTotal(0) {}
};

static __device__ void 
	DTimingClearAndStart(DTimingCounter &ACounter);
static __device__ void 
	DTimingFinish(DTimingCounter &ACounter);
static __device__ void 
	DTimingInitialize();
static __device__ float 
	DTimingSeconds(DTimingCounter &ACounter);
static __device__ void 
	DTimingFinish(DTimingCounter &ACounter, char *AStdOutTimingDesc);
static __device__ void 
	DTimingStart(DTimingCounter &ACounter);

// implementation

__device__ void DTimingClearAndStart(DTimingCounter &ACounter)
{
	ACounter.CounterTotal = 0;
	ACounter.Counter = clock64();
}

__device__ void DTimingFinish(DTimingCounter &ACounter)
{
	__int64 LCounter = clock64();
	ACounter.CounterTotal += (LCounter - ACounter.Counter);
}

__device__ void DTimingInitialize()
{
//!!
}

__device__ float DTimingSeconds(DTimingCounter &ACounter)
{
	return ((float)(ACounter.CounterTotal) / DEVICE_CLOCK_FREQUENCY);
}

__device__ void DTimingFinish(DTimingCounter &ACounter, char *AStdOutTimingDesc)
{
	DTimingFinish(ACounter);
	printf("Finished in %.9f sec(s)  - %s\n", DTimingSeconds(ACounter), AStdOutTimingDesc);
}

__device__ void DTimingStart(DTimingCounter &ACounter)
{
	ACounter.Counter = clock64();
}

#endif