.PHONY: all qcadesigner-cpu qcadesigner-cuda usage clean clean-all

.DEFAULT: usage

usage:
	@echo "make [all|qcadesigner-cpu|qcadesigner-cuda|usage|clean|clean-all] [CPU_OPT=<O1|O2|O3>|CPU_DBG=<-g|(-D symbol )+>|GPU_ARCH=<sm_10|sm_11|sm_12|sm_13|sm_20>|GPU_OPT=<--fast_math>|GPU_PREC=<FLOAT>|GPU_DBG=<-g|(-D symbol )+>]"
	@echo "all: compile both versions of qcadesigner, for cpu and gpu. Valid options are CPU_OPT, CPU_DBG, GPU_ARCH, GPU_OPT, GPU_DBG."
	@echo "qcadesigner-cpu: compile only the CPU version of qcadesigner. Valid options are: CPU_OPT, CPU_DBG."
	@echo "qcadesigner-cuda: compile only the GPU version of qcadesigner. Valid options are: GPU_ARCH, GPU_OPT, GPU_DBG."
	@echo "usage: show help for this makefile."

all: qcadesigner-cpu qcadesigner-cuda

qcadesigner-cpu:
	@echo "Making CPU version of qcadesigner. Please stand by..."
	@make -f .Makefile_cpu EXTRA_PARAMS1=$(CPU_OPT) EXTRA_PARAMS2=$(CPU_DBG)
	@echo "done."

qcadesigner-cuda:
	@echo "Making GPU version of qcadesigner. Please stand by..."
	@make -f .Makefile_gpu EXTRA_PARAMS1=$(GPU_ARCH) EXTRA_PARAMS2=$(GPU_OPT) EXTRA_PARAMS3=$(GPU_DBG) EXTRA_PARAMS4=$(GPU_PREC)
	@echo "done."
	
clean-all:
	@echo Removing object files and executables...
	@rm -f ../obj/*.o 
	@rm -f ../obj/objects/*.o 
	@rm -f ../obj/coloring/*.o 
	@rm -f ../obj/cuda/*.o 
	@rm -f ../bin/* 
	@rm -f *.o 
	@rm -f cuda/*.o 
	@rm -f objects/*.o 
	@rm -f coloring/*.o 
	@echo done. 

clean:
	@echo Removing object files ...
	@rm -f ../obj/*.o 
	@rm -f ../obj/objects/*.o 
	@rm -f ../obj/coloring/*.o 
	@rm -f ../obj/cuda/*.o 
	@rm -f *.o 
	@rm -f cuda/*.o 
	@rm -f objects/*.o 
	@rm -f coloring/*.o 
	@echo done.

