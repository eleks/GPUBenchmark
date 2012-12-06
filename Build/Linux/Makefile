CUDA_INC_PATH?=/usr/local/cuda/include
CUDA_LIB_PATH?=/usr/local/cuda/lib64

SOURCE_PATH = ../../Source/

CPP_FLAGS = -Xcompiler "-fPIC -O2 -g -std=gnu++0x"
CPP_FLAGS_COMPUTE_20 = -D CUDA -D CUDA42 -D OpenCL

CU_FLAGS = -use_fast_math -Xcompiler "-fPIC -O2 -g"
CU_FLAGS_COMPUTE_20 = -D CUDA -D CUDA42 -gencode=arch=compute_20,code=\"sm_20,compute_20\"

LINK_FLAGS = -lcuda -lcudart -lcurand -lopencl
LINK_FLAGS_COMPUTE_20 = 

CC = nvcc

CPP_FILES = \
	$(SOURCE_PATH)Common/CTools.cpp \
	$(SOURCE_PATH)Common/CUDATools.cpp \
	$(SOURCE_PATH)Common/OpenCLTools.cpp \
	$(SOURCE_PATH)GPUBenchmark.cpp \
	$(SOURCE_PATH)GPUBenchmarkCUDA.cpp \
	$(SOURCE_PATH)GPUBenchmarkOpenCL.cpp \
	$(SOURCE_PATH)Main.cpp

CU_FILES = \
	$(SOURCE_PATH)Common/CUDATools.cu \
	$(SOURCE_PATH)GPUBenchmark.cu

CPP_OBJECTS = $(CPP_FILES:.cpp=.o)
CU_OBJECTS = $(CU_FILES:.cu=_cu.o)

EXECUTABLE = GPUBenchmark
           
all:	$(CPP_FILES) $(CU_FILES) $(EXECUTABLE)

$(EXECUTABLE): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(CC) $(LINK_FLAGS) $(LINK_FLAGS_COMPUTE_20) $(addprefix COMPUTE_20/,$(notdir $(CPP_OBJECTS))) $(addprefix COMPUTE_20/,$(notdir $(CU_OBJECTS))) -o $(addprefix COMPUTE_20/,$@);

%.o: %.cpp BuildDirStructure
	$(CC) -c $(CPP_FLAGS) $(CPP_FLAGS_COMPUTE_20) $< -o $(addprefix COMPUTE_20/,$(notdir $@));

%_cu.o: %.cu BuildDirStructure
	$(CC) -c $(CU_FLAGS) $(CU_FLAGS_COMPUTE_20) $< -o $(addprefix COMPUTE_20/,$(notdir $@));

BuildDirStructure:
	mkdir -p COMPUTE_20;

clean:
	rm COMPUTE_20/*.o COMPUTE_20/$(EXECUTABLE)
