#ifdef OpenCL

#include <cstdlib>
#include <sstream>
#include "OpenCLTools.h"

void CLSafeCall(cl_int error, const char *flp, int line)
{
    if (error != CL_SUCCESS)
		throw new ECLException("OpenCL call failed", error, flp, line);
}

#endif