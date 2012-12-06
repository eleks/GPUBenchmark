#pragma once
 
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <string>
#include <sstream>
#include <exception>
#include <algorithm>

#ifdef _WIN32
#define NOMINMAX
#define OVERRIDE override
#include <windows.h>
#else
#define OVERRIDE
#include <errno.h>
#endif

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
	TypeName(const TypeName&); \
	TypeName& operator=(const TypeName&)

#define DESTRUCTOR_CATCH(stmts) try { stmts } catch (...) {}

//#define THROW(className, message) throw new className(message, -1, __FILE__, __LINE__)
//#define THROW(className, ...) throw new className(__VA_ARGS__, __FILE__, __LINE__)
#define THROW(className, message, errorCode) throw new className(message, errorCode, __FILE__, __LINE__)
#define NOT_IMPLEMENTED(name) throw new ENotImplemented(name, __FILE__, __LINE__)
#define NOT_SUPPORTED(name) throw new ENotSupported(name, __FILE__, __LINE__)
#define NOT_AVAILABLE(name) throw new ENotAvailable(name, __FILE__, __LINE__)

#ifndef _WIN32
typedef long long __int64;
typedef char byte;
#endif

#define DEFINE_EXCEPTION_CLASS(className) \
	class className : public ECustomException { \
	public: \
		explicit className(const char *errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : \
			ECustomException(errorMessage, errorCode, flp, line) {} \
		explicit className(std::string errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : \
			ECustomException(errorMessage, errorCode, flp, line) {} \
	}

class ECustomException : public std::exception
{
private:
	std::string _errorMessage;
	int _errorCode;
	const char *_flp;
	int _line;
public:
	explicit ECustomException(const char *errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : 
		_errorMessage(errorMessage), _errorCode(errorCode), _flp(flp), _line(line) {}
	explicit ECustomException(std::string errorMessage, int errorCode = -1, const char *flp = NULL, int line = 0) : 
		_errorMessage(errorMessage), _errorCode(errorCode), _flp(flp), _line(line) {}

	virtual ~ECustomException() throw() {}
	virtual const char * what() const throw() {
		return _errorMessage.c_str();
	}
	virtual std::string message() const {
		std::stringstream stream;
		stream << what() << " (Error Code: " << _errorCode;
		if (_flp)
			stream << ", file: " << _flp;
		if (_line)
			stream << ", line: " << _line;
		stream << ")";
		return stream.str();
	}
	int errorCode() { 
		return _errorCode;
	}
	int line() { 
		return _line;
	}
	const char * flp() { 
		return _flp;
	}
};

class ENotImplemented : public ECustomException
{
private:
	std::string _name;
public:
	explicit ENotImplemented(const char *name, const char *flp = NULL, int line = 0) : 
		ECustomException("not implemented", -1, flp, line), _name(name) {}
	virtual ~ENotImplemented() throw() {}
	virtual std::string message() const OVERRIDE {
		return _name + std::string(" ") + ECustomException::message();
	}
	std::string name() {
		return _name;
	}
};

class ENotSupported : public ECustomException
{
private:
	std::string _name;
public:
	explicit ENotSupported(const char *name, const char *flp = NULL, int line = 0) : 
		ECustomException("not supported", -1, flp, line), _name(name) {}
	virtual ~ENotSupported() throw() {}
	virtual std::string message() const OVERRIDE {
		return _name + std::string(" ") + ECustomException::message();
	}
	std::string name() {
		return _name;
	}
};

class ENotAvailable : public ECustomException
{
private:
	std::string _name;
public:
	explicit ENotAvailable(const char *name, const char *flp = NULL, int line = 0) : 
		ECustomException("not available", -1, flp, line), _name(name) {}
	virtual ~ENotAvailable() throw() {}
	virtual std::string message() const OVERRIDE {
		return _name + std::string(" ") + ECustomException::message();
	}
	std::string name() {
		return _name;
	}
};

struct TimingCounter {
	__int64 Counter;
	__int64 CounterTotal;

	TimingCounter() : Counter(0), CounterTotal(0) {}
};

enum CompareResult {crLess = -1, crEqual = 0, crGreater = 1};

std::string 
	BoolToStrYesNo(bool AValue);
CompareResult
	Compare(float v1, float v2);
void 
	IntToCharBufF(__int64 AValue, char *ABuf, size_t ASize);
std::string 
	IntToStrF(__int64 AValue);
std::string
	PathDirGet(const char *path);
std::string
	PathFLNGet(const char *path);
std::string 
	SysErrorMessage(int error);
int 
	SysLastErrorGet();
void 
	SysLastErrorThrow(const char * name);
void 
	SystemPause();
void 
	TimingClearAndStart(TimingCounter &ACounter);
void
	TimingFinish(TimingCounter &ACounter);
void 
	TimingFinish(TimingCounter &ACounter, const char *AStdOutTimingDesc);
void
	TimingInitialize();
double 
	TimingSeconds(TimingCounter &ACounter);
double 
	TimingSeconds();
void
	TimingStart(TimingCounter &ACounter);
std::string 
	WorkingDirGet();

template<class T> class DynArray;
template<class T> class DynArray2D;
template<class T> class DynArrayOfArray;
template<class T> class DynArrayOfArrayOfArray;

class AutoFile
{
	DISALLOW_COPY_AND_ASSIGN(AutoFile);
	FILE *_file;
	std::string _flp;
public:
	AutoFile() : _file(NULL) {};
	AutoFile(const char *flp, const char *mode) : _file(NULL) {
		int result = open(flp, mode);
		if (result != 0)
			THROW(ECustomException, std::string("Could not open file: ") + std::string(flp) + 
				std::string(". ") + SysErrorMessage(result), result);
	};
	~AutoFile() {
		DESTRUCTOR_CATCH(
			close();
		)
	};
	void close() {
		if (_file != NULL)
		{
			FILE *temp = _file;
			_file = NULL;
			_flp = "";
			fclose(temp); 
		}
	};
	int open(const char *flp, const char *mode) {
		close();
		_flp = flp;
		#ifdef _WIN32
			return fopen_s(&_file, flp, mode);
		#else
			_file = fopen(flp, mode);
			return _file == 0 ? errno : 0;
		#endif
	};
	void write(void *ptr, size_t size) {
		if (fwrite(ptr, size, 1, _file) != 1)
		{
			int result = SysLastErrorGet();
			THROW(ECustomException, std::string("Failed writing to file: ") + _flp + 
				std::string(". ") + SysErrorMessage(result), result);
		}
	}
	void read(void *ptr, size_t size) {
		if (fread(ptr, size, 1, _file) != 1)
		{
			int result = SysLastErrorGet();
			THROW(ECustomException, std::string("Failed reading from file: ") + _flp + 
				std::string(". ") + SysErrorMessage(result), result);
		}
	}
	int seek(int offset, int origin) { 
		return fseek(_file, offset, origin); 
	}
	int tell() {
		return ftell(_file);
	}
	FILE * file() {
		return _file;
	};
};

template<class T>
class DynArray
{
	DISALLOW_COPY_AND_ASSIGN(DynArray);
	T *_ptr;
	size_t _size;
	size_t _count;
public:
	DynArray() : _size(0), _count(0), _ptr(NULL) {};
	explicit DynArray(size_t aCount) : _ptr(NULL) {
		allocate(aCount);
	};
	~DynArray() {
		DESTRUCTOR_CATCH(
			release();
		)
	};
	void allocate(size_t count) {
		release();
		_count = count;
		_size = count * sizeof(T); 
		if (count > 0)
			_ptr = new T[count];
	};
	void release() {
		if (_ptr == NULL)
			return;
		delete [] _ptr;
		_ptr = NULL;
		_size = 0;
		_count = 0;
	}
	T& operator[](size_t index) {
		return _ptr[index];
	};
	void loadFromFile(const char *flp) {
		AutoFile file(flp, "rb");
		if (_ptr != NULL)
			file.read(_ptr, _size);
	}
	void saveToFile(const char *flp) {
		AutoFile file(flp, "wb");
		if (_ptr != NULL)
			file.write(_ptr, _size);
	}
	size_t size() {
		return _size;
	}
	size_t count() {
		return _count;
	}
	T* ptr() {
		return _ptr;
	}
};

template<class T>
class DynArray2D
{
	DISALLOW_COPY_AND_ASSIGN(DynArray2D);
	T *_ptr;
	size_t _size;
	size_t _dim1;
	size_t _dim2;
public:
	DynArray2D() : _size(0), _dim1(0), _dim2(0), _ptr(NULL) {};
	DynArray2D(size_t dim1, size_t dim2) : _ptr(NULL) {
		allocate(dim1, dim2);
	};
	~DynArray2D() {
		DESTRUCTOR_CATCH(
			release();
		)
	};
	void allocate(size_t dim1, size_t dim2) {
		release();
		_dim1 = dim1;
		_dim2 = dim2;
		_size = dim1 * dim2 * sizeof(T); 
		if (_size > 0)
			_ptr = new T[dim1 * dim2];
	};
	void release() {
		if (_ptr == NULL)
			return;
		delete [] _ptr;
		_ptr = NULL;
		_size = 0;
		_dim1 = 0;
		_dim2 = 0;
	}
	T& operator[](size_t index) {
		return _ptr[index];
	};
	T& operator()(size_t index1, size_t index2) {
		return _ptr[index1 * _dim2 + index2];
	};
	void loadFromFile(const char *flp) {
		AutoFile file(flp, "rb");
		if (_ptr != NULL)
			file.read(_ptr, _size);
	}
	void saveToFile(const char *flp) {
		AutoFile file(flp, "wb");
		if (_ptr != NULL)
			file.write(_ptr, _size);
	}
	T* ptr() {
		return _ptr;
	}
	size_t dim1() {
		return _dim1;
	}
	size_t dim2() {
		return _dim2;
	}
	size_t size() {
		return _size;
	}
};

template<class T>
class DynArrayOfArray
{
	DISALLOW_COPY_AND_ASSIGN(DynArrayOfArray);
	DynArray<DynArray<T> > _data;
public:
	DynArrayOfArray() {};
	DynArrayOfArray(size_t dim1, size_t dim2) {
		allocate(dim1, dim2);
	};
	~DynArrayOfArray() {
		DESTRUCTOR_CATCH(
			release();
		)
	};
	void allocate(size_t dim1, size_t dim2) {
		_data.allocate(dim1);
		for (size_t i = 0; i < dim1; i++)
			_data[i].allocate(dim2);
	}
	void release() {
		for (size_t i = 0; i < _data.count; i++)
			_data[i].release();
		_data.release();
	}
	DynArray<T>& operator[](size_t index) {
		return _data[index];
	};
};

template<class T>
class DynArrayOfArrayOfArray
{
	DISALLOW_COPY_AND_ASSIGN(DynArrayOfArrayOfArray);
	DynArray<DynArrayOfArray<T> > _data;
public:
	DynArrayOfArrayOfArray() {};
	DynArrayOfArrayOfArray(size_t dim1, size_t dim2, size_t dim3) {
		allocate(dim1, dim2, dim3);
	};
	~DynArrayOfArrayOfArray() {
		DESTRUCTOR_CATCH(
			release();
		)
	};
	void allocate(size_t dim1, size_t dim2, size_t dim3) {
		_data.allocate(dim1);
		for (size_t i = 0; i < dim1; i++)
			_data[i].allocate(dim2, dim3);
	}
	void release() {
		for (size_t i = 0; i < _data.count; i++)
			_data[i].release();
		_data.release();
	}
	DynArrayOfArray<T>& operator[](size_t index) {
		return _data[index];
	};
};
