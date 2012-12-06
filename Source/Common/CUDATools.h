#ifdef CUDA

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include <cuda_runtime_api.h>
#include <curand.h>
#include <cufft.h>

#include "CTools.h"

#define CUDA_TOOLS_THREADS_PER_BLOCK	128
#define CUDA_TOOLS_MAX_BLOCKS			256

#define CUDA_SAFECALL(error) cudaSafeCall(error, __FILE__, __LINE__)
#define CUFFT_SAFECALL(error) cufftSafeCall(error, __FILE__, __LINE__)
#define CURAND_SAFECALL(error) curandSafeCall(error, __FILE__, __LINE__)

void 
	cudaDevicePropertiesDump();
size_t 
	cudaFreeMemGet();
void 
	cudaMemInfoDump();
void 
	cudaSafeCall(cudaError error, const char *flp = NULL, int line = 0);
void 
	cufftSafeCall(cufftResult result, const char *flp = NULL, int line = 0);
void 
	curandSafeCall(curandStatus_t status, const char *flp = NULL, int line = 0);

DEFINE_EXCEPTION_CLASS(ECUDAException);
DEFINE_EXCEPTION_CLASS(ECUFFTException);
DEFINE_EXCEPTION_CLASS(ECURANDException);

template<class T> class deviceMem;
template<class T> class hostMem;
template<class T> class mappedMem;

template<class T>
class deviceMem
{
	DISALLOW_COPY_AND_ASSIGN(deviceMem);
	T *_dptr;
	size_t _size;
public:
	deviceMem() : _size(0), _dptr(NULL) {}
	deviceMem(size_t count) : _dptr(NULL) {
		allocate(count);
	}
	~deviceMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(size_t count) {
		release(); 
		_size = count * sizeof(T); 
		CUDA_SAFECALL(cudaMalloc((void**)&_dptr, _size));
	}
	void release() {
		if (_dptr == NULL)
			return;
		CUDA_SAFECALL(cudaFree(_dptr)); 
		_dptr = NULL; 
		_size = 0;
	}
	void clear() {
		CUDA_SAFECALL(cudaMemset(_dptr, 0, _size));
	}
	void copyFrom(T *hptr) {
		CUDA_SAFECALL(cudaMemcpy(_dptr, hptr, _size, cudaMemcpyHostToDevice));
	}
	void copyFrom(hostMem<T> &mem) {
		copyFrom(mem.hptr());
	}
	void copyFromAsync(T *hptr, cudaStream_t stream = 0) {
		CUDA_SAFECALL(cudaMemcpyAsync(_dptr, hptr, _size, cudaMemcpyHostToDevice, stream));
	}
	void copyFromAsync(hostMem<T> &mem, cudaStream_t stream = 0) {
		copyFromAsync(mem.hptr(), stream);
	}
	void copyTo(T *hptr) {
		CUDA_SAFECALL(cudaMemcpy(hptr, _dptr, _size, cudaMemcpyDeviceToHost));
	}
	void copyTo(hostMem<T> &mem) {
		copyTo(mem.hptr());
	}
	void copyTo(deviceMem<T> &mem) {
		CUDA_SAFECALL(cudaMemcpy(mem.dptr(), _dptr, _size, cudaMemcpyDeviceToDevice));
	}
	void copyToAsync(T *hptr, cudaStream_t stream = 0) {
		CUDA_SAFECALL(cudaMemcpyAsync(hptr, _dptr, _size, cudaMemcpyDeviceToHost, stream));
	}
	void copyToAsync(hostMem<T> &mem, cudaStream_t stream = 0) {
		copyToAsync(mem.hptr(), stream);
	}
	void copyToAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		CUDA_SAFECALL(cudaMemcpyAsync(mem.dptr(), _dptr, _size, cudaMemcpyDeviceToDevice, stream));
	}
	T* dptr() {
		return _dptr;
	}
	size_t size() {
		return _size;
	}
};

enum CUDA_HostMemType {cudahmtRegular = 0, cudahmtSpecial = 1};

template<class T>
class hostMem
{
	DISALLOW_COPY_AND_ASSIGN(hostMem);
	T *_hptr;
	size_t _size;
	unsigned int _flags;
	CUDA_HostMemType _type;
public:
	hostMem() : _type(cudahmtRegular), _size(0), _hptr(NULL) {}
	hostMem(size_t count, unsigned int flags) : _type(cudahmtSpecial), _flags(flags), _hptr(NULL) {
		allocate(count);
	}
	hostMem(size_t count) : _type(cudahmtRegular), _flags(0), _hptr(NULL) {
		allocate(count);
	}
	~hostMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(size_t count) {
		assert(_type == cudahmtRegular || _type == cudahmtSpecial);
		release();
		_size = count * sizeof(T); 
		if (_type == cudahmtRegular)
			_hptr = new T[count];
		else
			CUDA_SAFECALL(cudaHostAlloc((void**)&_hptr, _size, _flags));
	}
	void allocate(size_t count, unsigned int aFlags) {
		release();
		_type = cudahmtSpecial;
		_flags = aFlags;
		_size = count * sizeof(T); 
		CUDA_SAFECALL(cudaHostAlloc((void**)&_hptr, _size, _flags));
	}
	void release() {
		assert(_type == cudahmtRegular || _type == cudahmtSpecial);
		if (_hptr == NULL)
			return;
		if (_type == cudahmtRegular)
			delete [] _hptr;
		else
			CUDA_SAFECALL(cudaFreeHost(_hptr)); 
		_hptr = NULL;
		_size = 0;
	}
	T& operator[](size_t index) {
		return _hptr[index];
	}
	void copyFrom(deviceMem<T> &mem) {
		CUDA_SAFECALL(cudaMemcpy(_hptr, mem.dptr(), _size, cudaMemcpyDeviceToHost));
	}
	void copyFromAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		CUDA_SAFECALL(cudaMemcpyAsync(_hptr, mem.dptr(), _size, cudaMemcpyDeviceToHost, stream));
	}
	void copyTo(deviceMem<T> &mem) {
		CUDA_SAFECALL(cudaMemcpy(mem.dptr(), _hptr, _size, cudaMemcpyHostToDevice));
	}
	void copyToAsync(deviceMem<T> &mem, cudaStream_t stream = 0) {
		CUDA_SAFECALL(cudaMemcpyAsync(mem.dptr(), _hptr, _size, cudaMemcpyHostToDevice, stream));
	}
	void loadFromFile(const char *flp) {
		AutoFile file(flp, "rb");
		if (_hptr != NULL)
			file.read(_hptr, _size);
	}
	void saveToFile(const char *flp) {
		AutoFile file(flp, "wb");
		if (_hptr != NULL)
			file.write(_hptr, _size);
	}
	T * hptr() {
		return _hptr;
	}
	size_t size() {
		return _size;
	}
};

enum CUDA_MappedMemType {cudammtNone = 0, cudammtAlloc = 1, cudammtRegisterHost = 2};

template<class T>
class mappedMem
{
	DISALLOW_COPY_AND_ASSIGN(mappedMem);
	T *_hptr;
	T *_dptr;
	size_t _size;
	CUDA_MappedMemType _type;
public:
	mappedMem() : _size(0), _hptr(NULL), _dptr(NULL), _type(cudammtNone) {}
	mappedMem(size_t count) : _hptr(NULL), _dptr(NULL), _type(cudammtNone) {
		allocate(count);
	}
	~mappedMem() {
		DESTRUCTOR_CATCH(
			release();
		)
	}
	void allocate(size_t count) {
		release(); 
		_size = count * sizeof(T); 
		_type = cudammtAlloc;
		CUDA_SAFECALL(cudaHostAlloc((void**)&_hptr, _size, cudaHostAllocMapped)); 
		CUDA_SAFECALL(cudaHostGetDevicePointer((void**)&_dptr, _hptr, 0));
	}
	void registerHost(T *hptr, size_t count) {
		release(); 
		_size = count * sizeof(T); 
		_type = cudammtRegisterHost;
		CUDA_SAFECALL(cudaHostRegister(hptr, _size, cudaHostRegisterMapped)); 
		_hptr = hptr;
		CUDA_SAFECALL(cudaHostGetDevicePointer((void**)&_dptr, _hptr, 0));
	}
	void release() {
		if (_type == cudammtNone)
			return;
		if (_hptr == NULL)
		{
			_type = cudammtNone;
			return;
		}
		assert(_type == cudammtAlloc || _type == cudammtRegisterHost);
		if (_type == cudammtAlloc) 
			CUDA_SAFECALL(cudaFreeHost(_hptr));
		if (_type == cudammtRegisterHost)
			CUDA_SAFECALL(cudaHostUnregister(_hptr));
		_type = cudammtNone;
		_hptr = NULL; 
		_dptr = NULL;
		_size = 0;
	}
	T& operator[](size_t index) {return _hptr[index];};
	T * hptr() {
		return _hptr;
	}
	T * dptr() {
		return _dptr;
	}
	size_t size() {
		return _size;
	}
};

template<class T> void cuda_arraySub(T *A, T b, int count);
template<class T> void cuda_arrayMax(T *R, T *A, T b, int count);
template<class T> void cuda_arraySum(T *R, T *A, int count);
template<class T> void cuda_arrayStd(T *R, T *A, int count);

#endif