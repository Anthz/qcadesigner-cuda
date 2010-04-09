CC=gcc-4.3

CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)
NVCC=$(CUDA_HOME)/bin/nvcc

NVCFLAGS=  \
	-I $(CUDASDKROOT)/C/common/inc \
	-I $(CUDAROOT)/include

CFLAGS= \
	 -L$(CUDAROOT)/lib64  \
	 -L$(CUDASDKROOT)/C/lib

OBJECTS =\
		GlobalVSTexture_kernel.o\
      GlobalVSTexture_main.o 
      
	
GlobalVSTexture: $(OBJECTS)
	$(CC) $(CFLAGS)  -o $@ $^ -lcuda -lcudart -lcutil_x86_64

clean:
	rm -f *.o *.linkinfo GlobalVSTexture

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

