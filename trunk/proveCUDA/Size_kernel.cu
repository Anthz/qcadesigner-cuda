#include<cutil_inline.h>

__global__ void kernel (double* gpu_size) {

	double pippo = 1.0;

	pippo = pippo + 1;
	pippo = pippo* pippo;
	pippo = cos(pippo + 20.0);
	pippo = 1 / pippo;

   gpu_size[0] = sizeof(short);
   gpu_size[1] = sizeof(int);
   gpu_size[2] = sizeof(float);
   gpu_size[3] = sizeof(double);
   gpu_size[4] = pippo;   
	
}


extern "C"
void launchKernel (double* h_gpu_size) {
   
   double* d_gpu_size;
   dim3 threads(1);
   dim3 grid(1);

   cudaSetDevice( cutGetMaxGflopsDeviceId() );
   cutilSafeCall( cudaMalloc((void**) &d_gpu_size, 5*sizeof(double)) ); 

   kernel<<<threads, grid>>> (d_gpu_size);

   cudaThreadSynchronize();

   cutilSafeCall( cudaMemcpy(h_gpu_size, d_gpu_size, 5*sizeof(double), cudaMemcpyDeviceToHost) );

}
