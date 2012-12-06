#ifdef OpenCL

#pragma once

#include <cstdlib>
#include <CL/cl.h>
#include "CTools.h"

#define CL_SAFECALL(error) CLSafeCall(error, __FILE__, __LINE__)

void CLSafeCall(cl_int error, const char *flp = NULL, int line = 0);

class ECLException : public ECustomException
{
public:
	explicit ECLException(const char *errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : 
		ECustomException(errorMessage, errorCode, flp, line) {}
	explicit ECLException(std::string errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : 
		ECustomException(errorMessage, errorCode, flp, line) {}
};

class CLContext
{
private:
	cl_platform_id _platform;
	cl_device_id _device;
	cl_context _context;

public:
	CLContext() : _platform(0), _device(0), _context(0) {}
	~CLContext() {
		DESTRUCTOR_CATCH(
			finalize();
		)
	}
	void initialize(cl_device_type deviceType = CL_DEVICE_TYPE_GPU) {
		finalize();
		CL_SAFECALL(clGetPlatformIDs(1, &_platform, NULL));
		CL_SAFECALL(clGetDeviceIDs(_platform, deviceType, 1, &_device, NULL));
		cl_int error;
		_context = clCreateContext(0, 1, &_device, NULL, NULL, &error);
		CL_SAFECALL(error);
	}
	void finalize() {
		if (_context)
		{
			CL_SAFECALL(clReleaseContext(_context));
			_context = 0;
		}
	}
	cl_platform_id platform() { return _platform; }
	cl_device_id device() { return _device; }
	cl_context context() { return _context; }
};

struct CLCommandQueue
{
private:
	cl_command_queue _commandQueue;

public:
	CLCommandQueue() : _commandQueue(0) {}
	~CLCommandQueue() {
		DESTRUCTOR_CATCH(
			finalize();
		)
	}
	void initialize(CLContext &context) {
		cl_int error;
		_commandQueue = clCreateCommandQueue(context.context(), context.device(), 0, &error);
		CL_SAFECALL(error);
	}
	void finalize() {
		if (_commandQueue)
		{
			CL_SAFECALL(clReleaseCommandQueue(_commandQueue));
			_commandQueue = 0;
		}
	}
/*
	void barrier() {
		CL_SAFECALL(clEnqueueBarrier(_commandQueue));
	}
*/
	void finish() {
		CL_SAFECALL(clFinish(_commandQueue));
	}
	void flush() {
		CL_SAFECALL(clFlush(_commandQueue));
	}
	cl_command_queue commandQueue() { return _commandQueue; };
};

class CLProgram
{
private:
	cl_program _program;
	CLContext *_context;

public:
	CLProgram() : _program(0), _context(0) {}
	~CLProgram() {
		DESTRUCTOR_CATCH(
			finalize();
		)
	}
	void initialize(CLContext &context, std::string source) {
		_context = &context;
		const char *clSource = source.c_str();
		size_t clSourceLength = source.size();
		cl_int error;
		_program = clCreateProgramWithSource(context.context(), 1, 
			&clSource, &clSourceLength, &error);
		CL_SAFECALL(error);
	}
	void finalize() {
		if (_program)
		{
			CL_SAFECALL(clReleaseProgram(_program));
			_program = 0;
		}
	}
	void build(const char *flags) {
		assert(_context);
		cl_int error = clBuildProgram(_program, 0, NULL, flags, NULL, NULL);
		if (error == CL_SUCCESS)
			return;
		std::string buildLog;
		size_t buildLogMaxSize = 32768, buildLogSize = 0; 
		buildLog.resize(buildLogMaxSize);
		CL_SAFECALL(clGetProgramBuildInfo(_program, _context->device(), CL_PROGRAM_BUILD_LOG, 
			buildLogMaxSize, &buildLog[0], &buildLogSize));
		buildLog.resize(buildLogSize);
		std::string message("OpenCL Build Program failed");
		if (buildLogSize > 0)
			message.append(std::string(". BuildLog:\n") + buildLog);
		THROW(ECLException, message, error);
	}
	cl_program program() { return _program; }
};

class CLKernelBase
{
	DISALLOW_COPY_AND_ASSIGN(CLKernelBase);
public:
	CLKernelBase() {};
	virtual void finalize() = 0;
	virtual void initialize(CLProgram &program, const char * name) = 0;
	virtual void argSet(cl_uint index, size_t size, const void *value)  = 0;
	virtual void argSetInt(cl_uint index, int value) = 0;
	virtual void argSetInt64(cl_uint index, __int64 value) = 0;
	virtual void argSetFloat(cl_uint index, float value) = 0;
	virtual void argSetDouble(cl_uint index, double value) = 0;
	virtual void argSetPtr(cl_uint index, void *value) = 0;
	virtual void argSetMem(cl_uint index, cl_mem mem) = 0;
	virtual void argSetSharedMem(cl_uint index, size_t size) = 0;
	virtual cl_event enqueue1D(CLCommandQueue &commandQueue, size_t globalWorkSize, size_t localWorkSize) = 0;
	virtual void waitForCompletion() = 0;
	virtual cl_kernel kernel() = 0;
};

class CLKernelNA : public CLKernelBase
{
public:
	CLKernelNA() {};
	void finalize() { NOT_AVAILABLE("CLKernel"); }
	void initialize(CLProgram &program, const char * name) { NOT_AVAILABLE("CLKernel"); }
	void argSet(cl_uint index, size_t size, const void *value) { NOT_AVAILABLE("CLKernel"); }
	void argSetInt(cl_uint index, int value) { NOT_AVAILABLE("CLKernel"); }
	void argSetInt64(cl_uint index, __int64 value) { NOT_AVAILABLE("CLKernel"); }
	void argSetFloat(cl_uint index, float value) { NOT_AVAILABLE("CLKernel"); }
	void argSetDouble(cl_uint index, double value) { NOT_AVAILABLE("CLKernel"); }
	void argSetPtr(cl_uint index, void *value) { NOT_AVAILABLE("CLKernel"); }
	void argSetMem(cl_uint index, cl_mem mem) { NOT_AVAILABLE("CLKernel"); }
	void argSetSharedMem(cl_uint index, size_t size) { NOT_AVAILABLE("CLKernel"); }
	cl_event enqueue1D(CLCommandQueue &commandQueue, size_t globalWorkSize, size_t localWorkSize) { NOT_AVAILABLE("CLKernel"); }
	void waitForCompletion() { NOT_AVAILABLE("CLKernel"); }
	cl_kernel kernel() { NOT_AVAILABLE("CLKernel"); }
};

class CLKernel : public CLKernelBase
{
	cl_kernel _kernel;
	cl_event _event;
public:
	CLKernel() : _kernel(0), _event(0) {};
	~CLKernel() {
		DESTRUCTOR_CATCH(
			finalize();
		)
	}
	void finalize()	OVERRIDE {
		if (_kernel)
		{
			CL_SAFECALL(clReleaseKernel(_kernel));
			_kernel = 0;
		}
	}
	void initialize(CLProgram &program, const char * name) OVERRIDE {
		cl_int error;
	    _kernel = clCreateKernel(program.program(), name, &error);
		CL_SAFECALL(error);
	}
	void argSet(cl_uint index, size_t size, const void *value) OVERRIDE {
		CL_SAFECALL(clSetKernelArg(_kernel, index, size, value));
	}
	void argSetInt(cl_uint index, int value) OVERRIDE {
		argSet(index, sizeof(value), (void*)&value);
	}
	void argSetInt64(cl_uint index, __int64 value) OVERRIDE {
		argSet(index, sizeof(value), (void*)&value);
	}
	void argSetFloat(cl_uint index, float value) OVERRIDE {
		argSet(index, sizeof(value), (void*)&value);
	}
	void argSetDouble(cl_uint index, double value) OVERRIDE {
		argSet(index, sizeof(value), (void*)&value);
	}
	void argSetPtr(cl_uint index, void *value) OVERRIDE {
		argSet(index, sizeof(value), (void*)&value);
	}
	void argSetMem(cl_uint index, cl_mem mem) OVERRIDE {
		argSet(index, sizeof(cl_mem), (void*)&mem);
	}
	void argSetSharedMem(cl_uint index, size_t size) OVERRIDE {
		argSet(index, size, NULL);
	}
	cl_event enqueue1D(CLCommandQueue &commandQueue, size_t globalWorkSize, size_t localWorkSize) OVERRIDE {
		CL_SAFECALL(clEnqueueNDRangeKernel(commandQueue.commandQueue(), _kernel, 1, NULL, 
			&globalWorkSize, &localWorkSize, 0, NULL, &_event));
		return _event;
	}
	void waitForCompletion() OVERRIDE {
		CL_SAFECALL(clWaitForEvents(1, &_event));
	}
	cl_kernel kernel() OVERRIDE { 
		return _kernel; 
	};
};

template<class T> class CLDeviceMem;
template<class T> class CLHostMem;
template<class T> class CLMappedMem;

template<class T>
class CLDeviceMem
{
	DISALLOW_COPY_AND_ASSIGN(CLDeviceMem);
	cl_mem _mem;
	size_t _size;
	CLCommandQueue *_commandQueue;
	cl_event _event;
public:
	CLDeviceMem() : _mem(0), _size(0), _commandQueue(NULL) {};
	CLDeviceMem(CLContext &context, CLCommandQueue &commandQueue, size_t count) 
		: _mem(0), _size(0), _commandQueue(NULL) {
		allocate(context, commandQueue, count);
	}
	~CLDeviceMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(CLContext &context, CLCommandQueue &commandQueue, size_t count, cl_mem_flags flags = CL_MEM_READ_WRITE) {
		release(); 
		_commandQueue = &commandQueue;
		_size = count * sizeof(T); 
		cl_int error;
		_mem = clCreateBuffer(context.context(), flags, _size, NULL, &error);
		CL_SAFECALL(error);
	}
	void release() {
		if (_mem == NULL)
			return;
		CL_SAFECALL(clReleaseMemObject(_mem)); 
		_mem = NULL; 
		_size = 0;
		_commandQueue = NULL;
	}
	void clear() {
		NOT_IMPLEMENTED("CLDeviceMem::clear");
	}
	void copyFrom(T *hptr) {
		assert(_commandQueue);
		CL_SAFECALL(clEnqueueWriteBuffer(_commandQueue->commandQueue(), 
			_mem, CL_TRUE, 0, _size, hptr, 0, NULL, NULL));
	}
	void copyFrom(CLHostMem<T> &mem) {
		copyFrom(mem.hptr());
	}
	void copyFromAsync(T *hptr) {
		assert(_commandQueue);
		CL_SAFECALL(clEnqueueWriteBuffer(_commandQueue->commandQueue(), 
			_mem, CL_FALSE, 0, _size, hptr, 0, NULL, &_event));
	}
	void copyFromAsync(CLHostMem<T> &mem) {
		copyFromAsync(mem.hptr);
	}
	void copyTo(T *hptr) {
		assert(_commandQueue);
		CL_SAFECALL(clEnqueueReadBuffer(_commandQueue->commandQueue(), 
			_mem, CL_TRUE, 0, _size, hptr, 0, NULL, NULL));
	}
	void copyTo(CLHostMem<T> &mem) {
		copyTo(mem.hptr());
	}
	void copyToAsync(T *hptr) {
		assert(_commandQueue);
		CL_SAFECALL(clEnqueueReadBuffer(_commandQueue->commandQueue(), 
			_mem, CL_FALSE, 0, _size, hptr, 0, NULL, &_event));
	}
	void copyToAsync(CLHostMem<T> &mem) {
		copyToAsync(mem.hptr);
	}
	void copyToAsync(CLDeviceMem<T> &mem) {
		assert(_commandQueue);
		CL_SAFECALL(clEnqueueCopyBuffer(_commandQueue->commandQueue(), 
			_mem, mem.mem(), 0, 0, _size, 0, NULL, &_event));
	}
	void waitForCompletion() {
		CL_SAFECALL(clWaitForEvents(1, &_event));
	}
	cl_mem mem() { 
		return _mem; 
	}
};

enum CL_HostMemType {clhmtRegular = 0, clhmtSpecial = 1};

template<class T>
class CLHostMem
{
	DISALLOW_COPY_AND_ASSIGN(CLHostMem);
	T *_hptr;
	size_t _size;
	unsigned int _flags;
	CL_HostMemType _type;

public:
	CLHostMem() : _type(clhmtRegular), _size(0), _hptr(NULL) {};
	CLHostMem(size_t count, unsigned int flags) : _type(clhmtSpecial), _flags(flags), _hptr(NULL) {
		allocate(count);
	}
	CLHostMem(size_t count) : _type(clhmtRegular), _flags(0), _hptr(NULL) {
		allocate(count);
	}
	~CLHostMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(size_t count) {
		assert(_type == clhmtRegular || _type == clhmtSpecial);
		release();
		_size = count * sizeof(T); 
		if (_type == clhmtRegular)
			_hptr = new T[count];
		else
			NOT_IMPLEMENTED("CLHostMem::allocate clhmtSpecial");
			//cudaSafeCall(cudaHostAlloc((void**)&hptr, size, flags));
	}
	void allocate(size_t count, unsigned int aFlags) {
		release();
		_type = clhmtSpecial;
		_flags = aFlags;
		_size = count * sizeof(T); 
		NOT_IMPLEMENTED("CLHostMem::allocate clhmtSpecial");
		//cudaSafeCall(cudaHostAlloc((void**)&hptr, size, flags));
	}
	void release() {
		assert(_type == clhmtRegular || _type == clhmtSpecial);
		if (_hptr == NULL)
			return;
		if (_type == clhmtRegular)
			delete [] _hptr;
		else
			NOT_IMPLEMENTED("CLHostMem::release clhmtSpecial");
			//cudaSafeCall(cudaFreeHost(hptr)); 
		_hptr = NULL;
		_size = 0;
	}
	T& operator[](size_t index) {
		return _hptr[index];
	}
	void copyFrom(CLDeviceMem<T> &mem) {
		mem.copyTo(_hptr);
	}
	void copyFromAsync(CLDeviceMem<T> &mem) {
		mem.copyToAsync(_hptr);
	}
	void copyTo(CLDeviceMem<T> &mem) {
		mem.copyFrom(_hptr);
	}
	void copyToAsync(CLDeviceMem<T> &mem) {
		mem.copyFromAsync(_hptr);
	}
	T* hptr() {
		return _hptr;
	}
};

enum CL_MappedMemType {clmmtNone = 0, clmmtAlloc = 1, clmmtRegisterHost = 2};

template<class T>
class CLMappedMem
{
	DISALLOW_COPY_AND_ASSIGN(CLMappedMem);
	T *_hptr;
	cl_mem _mem;
	size_t _size;
	CL_MappedMemType _type;
	CLCommandQueue *_commandQueue;
public:
	CLMappedMem() : _size(0), _hptr(NULL), _mem(NULL), _type(clmmtNone) {};
	CLMappedMem(size_t count) : _hptr(NULL), _mem(NULL), _type(clmmtNone) {
		allocate(count);
	}
	~CLMappedMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(CLContext &context, CLCommandQueue &commandQueue, size_t count) {
		release();
		_commandQueue = &commandQueue;
		_size = count * sizeof(T);
		_type = clmmtAlloc;
		cl_int error;
		_mem = clCreateBuffer(context.context(), CL_MEM_ALLOC_HOST_PTR || CL_MEM_READ_WRITE, _size, NULL, &error);
		CL_SAFECALL(error);
	}
	void map() {
		assert(_commandQueue);
		cl_int error;
		_hptr = (T*)clEnqueueMapBuffer(_commandQueue->commandQueue(), _mem, CL_TRUE, 
			CL_MAP_READ || CL_MAP_WRITE, 0, _size, 0, NULL, NULL, &error);
		CL_SAFECALL(error);
	}
	void unmap() {
		assert(_commandQueue);
		assert(_hptr);
		cl_event eCompleted;
		CL_SAFECALL(clEnqueueUnmapMemObject(_commandQueue->commandQueue(), _mem, _hptr, 0, NULL, &eCompleted));
		_hptr = NULL;
		CL_SAFECALL(clWaitForEvents(1, &eCompleted));
	}
	void registerHost(CLContext &context, T *hptr, size_t count) {
		release(); 
		_size = count * sizeof(T); 
		_type = clmmtRegisterHost;
		cl_int error;
		_mem = clCreateBuffer(context.context(), CL_MEM_USE_HOST_PTR, _size, hptr, &error);
		CL_SAFECALL(error);
	}
	void release() {
		if (_type == clmmtNone)
			return;
		if (!_mem)
		{
			_type = clmmtNone;
			return;
		}
		assert(_type == clmmtAlloc || _type == clmmtRegisterHost);
		if (_type == clmmtAlloc && _hptr) 
			unmap();
		CL_SAFECALL(clReleaseMemObject(_mem));
		_type = clmmtNone;
		_hptr = NULL; 
		_mem = NULL;
		_size = 0;
	}
	T& operator[](size_t index) {return _hptr[index];};
	T * hptr() {
		return _hptr;
	}
	cl_mem mem() {
		return _mem;
	}
	size_t size() {
		return _size;
	}
};

#endif