#include <cutil_inline.h>
#include <cuda.h>

#include <stdio.h>
#include <math.h>

#include "cuPrintf.cu"

#define BLOCK_DIM 384


// declare texture reference for 1D float texture
texture<float, 1, cudaReadModeElementType> tex;


__global__ void textureKernel (float* result, int* neighbours, int result_dim, int neighbours_dim)
{

 if ( blockIdx.x*blockDim.x + threadIdx.x < result_dim ) {
   
      int i, num_neighbours;
      float my_polarization, neighbours_polarization; 

      num_neighbours = neighbours_dim / result_dim;

      my_polarization = tex1D(tex, blockIdx.x*blockDim.x + threadIdx.x);

      neighbours_polarization = 0;
      for (i = 0; i < num_neighbours; i++)
	 neighbours_polarization += tex1D(tex, neighbours[blockIdx.x*blockDim.x + threadIdx.x*num_neighbours + i]);

      result[blockIdx.x*blockDim.x + threadIdx.x] = my_polarization * neighbours_polarization;

   }

}


__global__ void globalKernel (float *result, float *polarization, int *neighbours, int result_dim, int polarization_dim, int neighbours_dim)
{

   int index = blockIdx.x*blockDim.x + threadIdx.x;
   int i, num_neighbours, index_next;
   float my_polarization, neighbours_polarization = 0; 

   //cuPrintf("Index: %d\n", index, polarization_dim);

   if ( index < polarization_dim ) {
      
      num_neighbours = neighbours_dim / polarization_dim;

      //cuPrintf("Number Neigh.: %d\n", num_neighbours);

      my_polarization = polarization[index];

      //cuPrintf("My polarization: %f\n", my_polarization);

      for (i = 0; i < num_neighbours; i++)
      {
         index_next = neighbours[index * num_neighbours + i];
         //cuPrintf("Next neighbour index: %d\n", index_next);

	 neighbours_polarization += polarization[index_next];
         //cuPrintf("Next neighbour pol: %f\n", polarization[index_next]);
      }

      result[index] = my_polarization * neighbours_polarization;

      //cuPrintf("Result: %f\n", result[index]);

   }


}


extern "C"
void launchGlobalKernel (float *h_result, float *h_polarization, int *h_neighbours, int h_result_dim, int h_polarization_dim, int h_neighbours_dim) {

   // Declarations
   float milliseconds;
   cudaEvent_t startEvent, stopEvent;
   float *d_polarization, *d_result;
   int *d_neighbours;	

   // Setup execution parameters
   dim3 threads(BLOCK_DIM);
   dim3 grid(ceil ((float)h_polarization_dim/BLOCK_DIM));

   printf("threads per block: %d\nblocks per grid: %d\n", threads.x, grid.x);

   cudaSetDevice( cutGetMaxGflopsDeviceId() );
   cudaPrintfInit ();

   cutilSafeCall(cudaMalloc((void**) &d_result, h_result_dim*sizeof(float))); 
   cutilSafeCall(cudaMalloc((void**) &d_polarization, h_polarization_dim*sizeof(float))); 
   cutilSafeCall(cudaMalloc((void**) &d_neighbours, h_neighbours_dim*sizeof(int))); 

   cutilSafeCall(cudaMemcpy(d_polarization, h_polarization, h_polarization_dim*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy(d_neighbours, h_neighbours, h_neighbours_dim*sizeof(int), cudaMemcpyHostToDevice));

   printf("Inizio kernel...\n");

   cutilSafeCall( cudaEventCreate(&startEvent) );
   cutilSafeCall( cudaEventCreate(&stopEvent) );
   cutilSafeCall(cudaEventRecord(startEvent, 0));

   // Execute the kernel
   globalKernel<<< grid, threads >>> (d_result, d_polarization, d_neighbours, h_result_dim, h_polarization_dim, h_neighbours_dim);

   // Wait Device
   cudaThreadSynchronize();

   cutilSafeCall(cudaEventRecord(stopEvent, 0));  
   cutilSafeCall(cudaEventSynchronize(stopEvent));
   cutilSafeCall(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
   
   printf("Fine kernel\n");

   cudaPrintfDisplay(stdout, true);

   cudaPrintfEnd();
   
   cutilSafeCall(cudaMemcpy(h_result, d_result, h_result_dim*sizeof(float), cudaMemcpyDeviceToHost));

   printf("Tempo d'esecuzione su GPU (Global Memory): %f ms.\n", milliseconds);

   cudaFree(d_result);
   cudaFree(d_polarization);
   cudaFree(d_neighbours);

}


extern "C"
void launchTextureKernel (float *h_result, float *h_polarization, int *h_neighbours, int h_result_dim, int h_polarization_dim, int h_neighbours_dim) {

   // Declarations
   float milliseconds;
   cudaEvent_t startEvent, stopEvent;
   float *d_result;
   int *d_neighbours;	

   // Setup execution parameters
   dim3 threads(BLOCK_DIM);
   dim3 grid(ceil(h_polarization_dim/BLOCK_DIM));

   cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // allocate device memory for result
   cutilSafeCall(cudaMalloc((void**) &d_result, h_result_dim*sizeof(float))); 
   cutilSafeCall(cudaMalloc((void**) &d_neighbours, h_neighbours_dim*sizeof(int))); 

   cutilSafeCall(cudaMemcpy(d_neighbours, h_neighbours, h_neighbours_dim*sizeof(int), cudaMemcpyHostToDevice));

   // allocate array and copy image data
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
   cudaArray* h_polarization_cu_array;
   cutilSafeCall( cudaMallocArray( &h_polarization_cu_array, &channelDesc, h_polarization_dim*sizeof(float) )); 
   cutilSafeCall( cudaMemcpyToArray( h_polarization_cu_array, 0, 0, h_polarization, h_polarization_dim*sizeof(float), cudaMemcpyHostToDevice));

   // set texture parameters
   tex.addressMode[0] = cudaAddressModeClamp;
   tex.filterMode = cudaFilterModePoint;
   tex.normalized = false;

   // Bind the array to the texture
   cutilSafeCall( cudaBindTextureToArray( tex, h_polarization_cu_array, channelDesc));

   printf("Inizio kernel...\n");

   cutilSafeCall( cudaEventCreate(&startEvent) );
   cutilSafeCall( cudaEventCreate(&stopEvent) );
   cutilSafeCall(cudaEventRecord(startEvent, 0));

   // Execute the kernel
   textureKernel<<< grid, threads >>> (d_result, d_neighbours, h_result_dim, h_neighbours_dim);

   // Wait Device
   cudaThreadSynchronize();

   cutilSafeCall(cudaEventRecord(stopEvent, 0));  
   cutilSafeCall(cudaEventSynchronize(stopEvent));
   cutilSafeCall(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
   
   printf("Fine kernel\n");

   cutilSafeCall(cudaMemcpy(h_result, d_result, h_result_dim*sizeof(float), cudaMemcpyDeviceToHost));

   printf("Tempo d'esecuzione su GPU (Texture Memory): %f ms.\n", milliseconds);


}


