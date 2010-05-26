CC = gcc-4.3
NVCC = $(CUDA_HOME)/bin/nvcc
GLIB=`pkg-config gtk+-2.0 --cflags --libs glib-2.0`

CFLAGS= \
	 -L$(CUDAROOT)/lib64  \
	 -L$(CUDASDKROOT)/C/lib \
	 -lcuda \
	 -lcudart \
	 -lcutil_x86_64 \
	 -O2
	 
CUDAROOT=$(CUDA_HOME)
CUDASDKROOT=$(CUDA_SDK_HOME)

OBJECTS= \
	main_batch_sim.o \
	fileio.o \
	bistable_simulation.o \
	coherence_vector.o \
	design.o \
	exp_array.o \
	fileio_helpers.o \
	generic_utils.o \
	simulation.o \
	simulation_data.o \
	vector_table.o \
	objects/object_helpers.o \
	objects/QCADCell.o \
	objects/QCADCompoundDO.o \
	objects/QCADDesignObject.o \
	objects/QCADDOContainer.o \
	objects/QCADLabel.o \
	objects/QCADLayer.o \
	objects/QCADStretchyObject.o \
	objects/QCADSubstrate.o \
	cuda/bistable_cuda.o \
	cuda/conversion.o 
	

NVCFLAGS= \
	-I $(CUDASDKROOT)/C/common/inc \
	-I $(CUDAROOT)/include \
	-arch=sm_11

all: QCADesigner

QCADesigner: $(OBJECTS)
	$(CC) $(GLIB) $(CFLAGS)  -o $@ $^
	
%.o: %.c
	$(CC) $(GLIB) $(CFLAGS) -c -o $@ $<
	
cuda/%.o: cuda/%.c
	$(CC) $(GLIB) $(CFLAGS) -c -o $@ $<
	
objects/%.o: objects/%.c
	$(CC) $(GLIB) $(CFLAGS) -c -o $@ $<
	
cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCFLAGS) -c -o $@ $<

clean: 
	rm -f *.o objects/*.o cuda/*.o
	rm QCADesigner
