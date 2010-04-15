CC=gcc-4.3

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
	coherence_vector_NO_PITCH.o \
 	test_coherence_vector.o 
      
	
test: $(OBJECTS)
	$(CC) $(CFLAGS)  -o $@ $^ -lcuda -lcudart -lcutil_x86_64

clean:
	rm -f *.o *.linkinfo test

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

