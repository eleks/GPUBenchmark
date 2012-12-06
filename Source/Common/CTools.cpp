#include "CTools.h"

static __int64 FPerformanceFrequency;

std::string BoolToStrYesNo(bool AValue)
{
	return std::string(AValue ? "Yes" : "No");
}

CompareResult Compare(float v1, float v2)
{
	return v1 < v2 ? crLess : v1 > v2 ? crGreater : crEqual;
}

void IntToCharBufF(__int64 AValue, char *ABuf, size_t ASize)
{
	const size_t MAX_BUF = 27;
	char LBuf[MAX_BUF], *LPBuf;
	LPBuf = LBuf + MAX_BUF - 1;
	*LPBuf-- = 0;
//	lldiv_t LValue = {AValue, 0};
	__int64 LQuot = AValue;
	char LRem = 0;
	int LCount = 0;
	if (LQuot == 0)
		*LPBuf-- = '0';
	else
		while (LQuot > 0)
		{
			LRem = LQuot % 10;
			LQuot = LQuot / 10;
			//LValue = lldiv(LValue.quot, 10);
			*LPBuf-- = '0' + LRem;
			if ((++LCount == 3) && (LQuot > 0))
			{
				*LPBuf-- = ',';
				LCount = 0;
			}
		}
	std::memcpy(ABuf, LPBuf + 1, std::min((size_t)(LBuf + MAX_BUF - LPBuf - 1), ASize));
}

std::string IntToStrF(__int64 AValue)
{
	char LResult[27];
	IntToCharBufF(AValue, LResult, 27);
	return std::string(LResult);
}

std::string PathDirGet(const char *path)
{
	std::string result(path);
	size_t len = result.size();
	while (len > 0 && result[len - 1] != '\\' && result[len - 1] != '/')
		len--;
	result.resize(len);
	return result;
}

std::string PathFLNGet(const char *path)
{
	std::string dir = PathDirGet(path);
	std::string fln(path + dir.size());
	return fln;
}

void TimingClearAndStart(TimingCounter &ACounter)
{
	ACounter.CounterTotal = 0;
	TimingStart(ACounter);
}

void TimingFinish(TimingCounter &ACounter)
{
	__int64 LCounter;
	#ifdef _WIN32
		QueryPerformanceCounter((LARGE_INTEGER*)&(LCounter));
	#else
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		LCounter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
	#endif
	ACounter.CounterTotal += (LCounter - ACounter.Counter);
}

void TimingFinish(TimingCounter &ACounter, const char *AStdOutTimingDesc)
{
	TimingFinish(ACounter);
	printf("Finished in %.6f sec(s)  - %s\n", TimingSeconds(ACounter), AStdOutTimingDesc);
}

void TimingInitialize()
{
	#ifdef _WIN32
		QueryPerformanceFrequency((LARGE_INTEGER*)&FPerformanceFrequency);
	#else
		FPerformanceFrequency = 1000000000; // nanosec
	#endif
}

double TimingSeconds()
{
	__int64 LCounter;
	#ifdef _WIN32
		QueryPerformanceCounter((LARGE_INTEGER*)&(LCounter));
	#else
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		LCounter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
	#endif
	return (double)LCounter / FPerformanceFrequency;
}

double TimingSeconds(TimingCounter &ACounter)
{
	return ((double)(ACounter.CounterTotal) / FPerformanceFrequency);
}

void TimingStart(TimingCounter &ACounter)
{
	#ifdef _WIN32
		QueryPerformanceCounter((LARGE_INTEGER*)&(ACounter.Counter));
	#else
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		ACounter.Counter = FPerformanceFrequency * ts.tv_sec + ts.tv_nsec;
	#endif
}
	
std::string SysErrorMessage(int error)
{
	const size_t maxLength = 1024;
	std::string result;
	result.resize(maxLength);
	#ifdef _WIN32
		int length = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS |
			FORMAT_MESSAGE_ARGUMENT_ARRAY, NULL, error, 0, &result[0], maxLength, NULL);
		if (length == 0)
			THROW(ECustomException, "FormatMessage failed", GetLastError());
		while (length > 0 && (result[length - 1] <= 32 || result[length - 1] == '.'))
			length--;
		result.resize(length);
	#else
		if (!strerror_r(error, &result[0], maxLength))
			throw new ECustomException("strerror_r", errno);
		result.resize(strlen(result.c_str()));
	#endif
	return result;
}

int SysLastErrorGet()
{
	#ifdef _WIN32
		return (int)GetLastError();
	#else
		return (int)errno;
	#endif
}

void SysLastErrorThrow(const char * name)
{
	int sysLastError = SysLastErrorGet();
	THROW(ECustomException, std::string(name) + std::string(" failed. ") + 
		SysErrorMessage(sysLastError), sysLastError);
}

void SystemPause()
{
	#ifdef _WIN32
		system("pause");
	#else
		printf("Press ENTER to continue ...\n");
		getchar();
	#endif
}

std::string WorkingDirGet()
{
	std::string result;
	#ifdef _WIN32
		DWORD size = GetCurrentDirectoryA(0, NULL);
		if (size == 0)
			SysLastErrorThrow("GetCurrentDirectory");
		result.resize(size - 1); // size includes terminating \0 character
		GetCurrentDirectoryA(size, &result[0]);
	#else
		size_t maxPathLen = 1024; /*MAXPATHLEN*/
		result.resize(maxPathLen);
        if (!getcwd(&result[0], maxPathLen));	
			SysLastErrorThrow("getcwd");
		result.resize(strlen(result.c_str()));
	#endif
	if (result.size() > 0 && result[result.size() - 1] != '\\')
		result.append("\\");
	return result;
}
