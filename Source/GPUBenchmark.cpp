#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "Common/CTools.h"
#include "GPUBenchmark.h"

#define MAX_TIMING_COUNT	5

double GPUBenchmark::execOuterTiming(benchmarkOuterFunc func, int p1, int p2, int p3, int p4, int p5)
{
	TimingCounter counter;
	double minTime = 1E10;
	for (int i = 0; i < MAX_TIMING_COUNT; i++)
	{
		TimingClearAndStart(counter);
		(this->*func)(p1, p2, p3, p4, p5);
		TimingFinish(counter);
		minTime = std::min(minTime, TimingSeconds(counter));
	}
	return minTime;
}

double GPUBenchmark::execInnerTiming(benchmarkInnerFunc func, int p1, int p2, int p3, int p4, int p5)
{
	TimingCounter counter;
	double minTime = 1E10;
	for (int i = 0; i < MAX_TIMING_COUNT; i++)
	{
		double time = (this->*func)(p1, p2, p3, p4, p5);
		minTime = std::min(minTime, time);
	}
	return minTime;
}

void GPUBenchmark::run()
{
	deviceInitialize();
	devicePropertiesShow();
	
	printf("***** Host kernel schedule latencies, microseconds\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelScheduleTinyTask, 16384, 128) * 1000000.0);
	printf("***** Host kernel execution latencies, microseconds\n");
	printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 1, 1) * 1000000.0);
	printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 1, 32) * 1000000.0);
	printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 8, 64) * 1000000.0);
	printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelExecuteTinyTask, 16384, 128) * 1000000.0);
	printf("\n");

#ifdef CUDA50
	if (isDynamicParallelismSupported())
	{
		printf("***** Device kernel schedule latencies\n");
		printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 1, 1) * 1000000.0);
		printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 1, 32) * 1000000.0);
		printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 8, 64) * 1000000.0);
		printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicScheduleTinyTask, 16384, 128) * 1000000.0);
		printf("***** Device kernel execution latencies\n");
		printf("1x1:                                              %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 1, 1) * 1000000.0);
		printf("1x32:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 1, 32) * 1000000.0);
		printf("8x64:                                             %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 8, 64) * 1000000.0);
		printf("16384x128:                                        %12.3f microsec\n", execInnerTiming(&GPUBenchmark::kernelDynamicExecuteTinyTask, 16384, 128) * 1000000.0);
		printf("\n");
	}
#endif

	printf("***** Reduction time (SUM), microseconds\n");
	printf("1x32,    256 elements, int:                       %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_int, 256, 10000, 1, 32) * 1000000.0);
	printf("1x32,    256 elements, float:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_float, 256, 10000, 1, 32) * 1000000.0);
	printf("1x32,    256 elements, double:                    %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_double, 256, 10000, 1, 32) * 1000000.0);
	if (isInt64Supported())
		printf("128x64,  1M elements, int64:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_int64, 1048576, 10, 128, 64) * 1000000.0);
	printf("128x64,  1M elements, float:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_float, 1048576, 10, 128, 64) * 1000000.0);
	printf("128x64,  1M elements, double:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_double, 1048576, 10, 128, 64) * 1000000.0);
	if (isInt64Supported())
		printf("128x256, 1M elements, int64:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_int64, 1048576, 10, 128, 256) * 1000000.0);
	printf("128x256, 1M elements, float:                      %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_float, 1048576, 10, 128, 256) * 1000000.0);
	printf("128x256, 1M elements, double:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_double, 1048576, 10, 128, 256) * 1000000.0);
	if (isInt64Supported())
		printf("512x512, 10M elements, int64:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_int64, 10485760, 1, 512, 512) * 1000000.0);
	printf("512x512, 10M elements, float:                     %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_float, 10485760, 1, 512, 512) * 1000000.0);
	printf("512x512, 10M elements, double:                    %12.3f microsec\n", execInnerTiming(&GPUBenchmark::reductionSum_double, 10485760, 1, 512, 512) * 1000000.0);
	printf("\n");

	printf("***** Dependent FLOP, GFLOPs\n");
	printf("ADD (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd_float, 16384, 512, 128) / 1E9);
	printf("MUL (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMul_float, 16384, 512, 128) / 1E9);
	printf("DIV (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelDiv_float, 16384, 512, 128) / 1E9);
	printf("SIN (512x128, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelSin_float, 16384, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd_double, 16384, 512, 128) / 1E9);
	printf("MUL (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMul_double, 16384, 512, 128) / 1E9);
	printf("DIV (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelDiv_double, 16384, 512, 128) / 1E9);
	if (isDoublePrecisionTrigonometrySupported())
		printf("SIN (512x128, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelSin_double, 16384, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd_float, 16384, 512, 512) / 1E9);
	printf("MUL (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMul_float, 16384, 512, 512) / 1E9);
	printf("DIV (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelDiv_float, 16384, 512, 512) / 1E9);
	printf("SIN (512x512, float):                             %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelSin_float, 16384, 512, 512) / 1E9);
	printf("\n");
	printf("ADD (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd_double, 16384, 512, 512) / 1E9);
	printf("MUL (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMul_double, 16384, 512, 512) / 1E9);
	printf("DIV (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelDiv_double, 16384, 512, 512) / 1E9);
	if (isDoublePrecisionTrigonometrySupported())
		printf("SIN (512x512, double):                            %12.3f GFLOPs\n", (double)16384 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelSin_double, 16384, 512, 512) / 1E9);
	printf("\n");

	printf("***** Independent FLOP, GFLOPs\n");
	printf("ADD (512x128, float):                             %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd_indep_float, 65536, 512, 128) / 1E9);
	printf("MAD (MulAdd) (512x128, float):                    %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMad_float, 65536, 512, 128) / 1E9);
//	printf("MAD+SF (MulAdd+SpecialFunc) (512x128, float):     %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMadSF_float, 65536, 512, 128) / 1E9);
	printf("ADD (512x128, double):                            %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelAdd_indep_double, 65536, 512, 128) / 1E9);
	printf("MAD (MulAdd) (512x128, double):                   %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMad_double, 65536, 512, 128) / 1E9);
//	printf("MAD+SF (MulAdd+SpecialFunc) (512x128, double):    %12.3f GFLOPs\n", (double)65536 * 512 * 128 / execOuterTiming(&GPUBenchmark::kernelMadSF_double, 65536, 512, 128) / 1E9);
	printf("\n");
	printf("ADD (512x512, float):                             %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd_indep_float, 65536, 512, 512) / 1E9);
	printf("MAD (MulAdd) (512x512, float):                    %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMad_float, 65536, 512, 512) / 1E9);
//	printf("MAD+SF (MulAdd+SpecialFunc) (512x512, float):     %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMadSF_float, 65536, 512, 512) / 1E9);
	printf("ADD (512x512, double):                            %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelAdd_indep_double, 65536, 512, 512) / 1E9);
	printf("MAD (MulAdd) (512x512, double):                   %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMad_double, 65536, 512, 512) / 1E9);
//	printf("MAD+SF (MulAdd+SpecialFunc) (512x512, double):    %12.3f GFLOPs\n", (double)65536 * 512 * 512 / execOuterTiming(&GPUBenchmark::kernelMadSF_double, 65536, 512, 512) / 1E9);
	printf("\n");

	printf("***** Memory management, milliseconds\n");
	printf("Device Memory allocate/release (16 bytes):        %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 16, 10) / 10 * 1000);
	printf("Mapped Memory allocate/release (16 bytes):        %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 16, 10) / 10 * 1000);
	printf("Host Memory register/unregister (16 bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 16, 10) / 10 * 1000);
	if (isWriteCombinedMemorySupported())
		printf("Host Write Combined allocate/release (16 bytes):  %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 16, 10) / 10 * 1000);
	printf("\n");
	printf("Device Memory allocate/release (10M bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 10485760, 1) / 1 * 1000);
	printf("Mapped Memory allocate/release (10M bytes):       %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 10485760, 1) / 1 * 1000);
	printf("Host Memory register/unregister (10M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 10485760, 1) / 1 * 1000);
	if (isWriteCombinedMemorySupported())
		printf("Host Write Combined allocate/release (10M bytes): %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 10485760, 1) / 1 * 1000);
	printf("\n");
	printf("Device Memory allocate/release (100M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::deviceMemAllocRelease, 104857600, 1) / 1 * 1000);
	printf("Mapped Memory allocate/release (100M bytes):      %12.3f millisec\n", execOuterTiming(&GPUBenchmark::mappedMemAllocRelease, 104857600, 1) / 1 * 1000);
	printf("Host Memory register/unregister (100M bytes):     %12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemRegisterUnregister, 104857600, 1) / 1 * 1000);
	if (isWriteCombinedMemorySupported())
		printf("Host Write Combined allocate/release (100M bytes):%12.3f millisec\n", execOuterTiming(&GPUBenchmark::hostMemWriteCombinedAllocRelease, 104857600, 1) / 1 * 1000);
	printf("\n");

	printf("***** Memory transfer speed (100MB blocks)\n");
	printf("Regular (Host to GPU):                            %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 0) / 1024 / 1024 / 1024);
	if (isPageLockedMemorySupported())
		printf("Page locked (Host to GPU):                        %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 1, 0) / 1024 / 1024 / 1024);
	if (isWriteCombinedMemorySupported())
		printf("Write Combined (Host to GPU):                     %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 2, 0) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Regular (GPU to Host):                            %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 1) / 1024 / 1024 / 1024);
	if (isPageLockedMemorySupported())
		printf("Page locked (GPU to Host):                        %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 1, 1) / 1024 / 1024 / 1024);
	if (isWriteCombinedMemorySupported())
		printf("Write Combined (GPU to Host):                     %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 2, 1) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Device (GPU to GPU):                              %12.3f GB/sec\n", (double)104857600.0 / execInnerTiming(&GPUBenchmark::memTransfer, 104857600, 0, 2) / 1024 / 1024 / 1024);
	printf("\n");

	printf("***** Device memory access speed (1024 x 512)\n");
	printf("Aligned read (int):                               %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (float):                             %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("Aligned read (double):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600, 10, 1024, 512, maAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Aligned write (int):                              %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("Aligned write (float):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("Aligned write (double):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600, 10, 1024, 512, maAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned read (int):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (float):                         %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("Not aligned read (double):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600, 1, 1024, 512, maNotAlignedRead) / 1024 / 1024 / 1024);
	printf("\n");
	printf("Not aligned write (int):                          %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (float):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("Not aligned write (double):                       %12.3f GB/sec\n", (double)104857600.0 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600, 1, 1024, 512, maNotAlignedWrite) / 1024 / 1024 / 1024);
	printf("\n");

	if (isPinnedMemorySupported())
	{
		printf("***** Pinned memory access speed (1024 x 512)\n");
		printf("Aligned read (int):                               %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
		printf("Aligned read (float):                             %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
		printf("Aligned read (double):                            %12.3f GB/sec\n", (double)104857600.0/2 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600/2, 1, 1024, 512, maPinnedAlignedRead) / 1024 / 1024 / 1024);
		printf("\n");
		printf("Aligned write (int):                              %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
		printf("Aligned write (float):                            %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
		printf("Aligned write (double):                           %12.3f GB/sec\n", (double)104857600.0/2 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600/2, 1, 1024, 512, maPinnedAlignedWrite) / 1024 / 1024 / 1024);
		printf("\n");
		printf("Not aligned read (int):                           %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
		printf("Not aligned read (float):                         %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
		printf("Not aligned read (double):                        %12.3f GB/sec\n", (double)104857600.0/2 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600/2, 1, 1024, 512, maPinnedNotAlignedRead) / 1024 / 1024 / 1024);
		printf("\n");
		printf("Not aligned write (int):                          %12.3f GB/sec\n", (double)104857600.0 * sizeof(int) / execInnerTiming(&GPUBenchmark::deviceMemAccess_int, 104857600, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
		printf("Not aligned write (float):                        %12.3f GB/sec\n", (double)104857600.0 * sizeof(float) / execInnerTiming(&GPUBenchmark::deviceMemAccess_float, 104857600, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
		printf("Not aligned write (double):                       %12.3f GB/sec\n", (double)104857600.0/2 * sizeof(double) / execInnerTiming(&GPUBenchmark::deviceMemAccess_double, 104857600/2, 1, 1024, 512, maPinnedNotAlignedWrite) / 1024 / 1024 / 1024);
		printf("\n");
	}
}