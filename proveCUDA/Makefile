CC=gcc

CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS= \
	-I $(CUDASDKROOT)/C/common/inc \
	-I $(CUDAROOT)/include

CFLAGS= \
	-L$(CUDAROOT)/lib64  \
	-L$(CUDASDKROOT)/C/lib

OBJECTS =\
	Size_kernel.o\
      	Size_main.o 
      
	
Size: $(OBJECTS)
	$(CC) $(CFLAGS)  -o $@ $^ -lcuda -lcudart -lcutil_x86_64

clean:
	rm -f *.o *.linkinfo Constant

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

