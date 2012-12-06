#include "Common/CTools.h"
#include "Common/CUDATools.h"
#include "Common/OpenCLTools.h"
#include "GPUBenchmarkCUDA.h"
#include "GPUBenchmarkOpenCL.h"

int usageShow()
{
	printf("Usage:   GPUBenchmark [CUDA|OpenCL]\n");
	printf("Example: GPUBenchmark CUDA\n");
	return 0;
}

int main(int argc, char * argv[])
{
	try
	{
		if (argc <= 1 || argc > 2)
			return usageShow();

		TimingInitialize();
		if (strcmp(argv[1], "OpenCL") == 0)
		{
			printf("Started GPUBenchmark using OpenCL\n");
			#ifdef OpenCL
				GPUBenchmarkCL benchmark(PathDirGet(argv[0]));
				benchmark.run();
			#else
				printf("ERROR: Not compiled with OpenCL define\n");
			#endif
		}
		else
		if (strcmp(argv[1], "CUDA") == 0)
		{
			printf("Started GPUBenchmark using CUDA\n");
			#ifdef CUDA
				GPUBenchmarkCUDA benchmark;
				benchmark.run();
			#else
				printf("ERROR: Not compiled with CUDA define\n");
			#endif
		} else
			return usageShow();
		printf("Completed successfully\n");
		SystemPause();
		return 0;
	}
	catch (ECustomException *E) 
	{
		printf("Exception: %s\n", E->message().c_str());
		SystemPause();
		return 1;
	}
	catch (std::exception *E) 
	{
		printf("Exception: %s\n", E->what());
		SystemPause();
		return 1;
	}
	catch (...) 
	{
		printf("Unknown exception occured\n");
		SystemPause();
		return SysLastErrorGet(); //!!
	}
}
