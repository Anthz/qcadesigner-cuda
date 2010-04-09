#include <cutil_inline.h>
#include <cuda.h>

#include <stdio.h>
#include <math.h>

#define BLOCK_DIM 128


// declare texture reference for 1D float texture
texture<float, 1, cudaReadModeElementType> tex;


__global__ void textureKernel (double* polarization, int* neighbours, int polarization_dim, int neighbours_dim)
{

 if ( blockIdx.x*blockDim.x + threadIdx.x < polarization_dim ) {
   
      int i, num_neighbours;
      double my_polarization, neighbours_polarization; 

      num_neighbours = neighbours_dim / polarization_dim;

      my_polarization = polarization[blockIdx.x*blockDim.x + threadIdx.x];

      neighbours_polarization = 0;
      for (i = 0; i < num_neighbours; i++)
	 neighbours_polarization += tex1D(tex, neighbours[blockIdx.x*blockDim.x + threadIdx.x*num_neighbours + i]);

      my_polarization *= neighbours_polarization;

      polarization[blockIdx.x*blockIdx.x + threadIdx.x] = my_polarization;

   }

}


__global__ void globalKernel (double* polarization, int* neighbours, int polarization_dim, int neighbours_dim)
{

   if ( blockIdx.x*blockDim.x + threadIdx.x < polarization_dim ) {
   
      int i, num_neighbours;
      double my_polarization, neighbours_polarization; 

      num_neighbours = neighbours_dim / polarization_dim;

      my_polarization = polarization[blockIdx.x*blockDim.x + threadIdx.x];

      neighbours_polarization = 0;
      for (i = 0; i < num_neighbours; i++)
	 neighbours_polarization += polarization[ neighbours[blockIdx.x*blockDim.x + threadIdx.x*num_neighbours + i] ];

      my_polarization *= neighbours_polarization;

      polarization[blockIdx.x*blockIdx.x + threadIdx.x] = my_polarization;

   }

}


extern "C"
void launchGlobalKernel (double* h_polarization, int* h_neighbours, int h_polarization_dim, int h_neighbours_dim) {

   // Declarations
   float milliseconds;
   cudaEvent_t startEvent, stopEvent;
   double* d_polarization;
   int* d_neighbours;	

   // Setup execution parameters
   dim3 threads(BLOCK_DIM);
   dim3 grid(ceil(h_polarization_dim/BLOCK_DIM));

   cudaSetDevice( cutGetMaxGflopsDeviceId() );

   cutilSafeCall(cudaMalloc((void**) &d_polarization, h_polarization_dim*sizeof(double))); 
   cutilSafeCall(cudaMalloc((void**) &d_neighbours, h_neighbours_dim*sizeof(int))); 

   cutilSafeCall(cudaMemcpy(d_polarization, h_polarization, h_polarization_dim*sizeof(double), cudaMemcpyHostToDevice));
   cutilSafeCall(cudaMemcpy(d_neighbours, h_neighbours, h_neighbours_dim*sizeof(int), cudaMemcpyHostToDevice));

   printf("Inizio kernel...\n");

   cutilSafeCall( cudaEventCreate(&startEvent) );
   cutilSafeCall( cudaEventCreate(&stopEvent) );
   cutilSafeCall(cudaEventRecord(startEvent, 0));

   // Execute the kernel
   globalKernel<<< grid, threads >>> (d_polarization, d_neighbours, h_polarization_dim, h_neighbours_dim);

   // Wait Device
   cudaThreadSynchronize();

   cutilSafeCall(cudaEventRecord(stopEvent, 0));  
   cutilSafeCall(cudaEventSynchronize(stopEvent));
   cutilSafeCall(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
   
   printf("Fine kernel\n");
   
   cutilSafeCall(cudaMemcpy(h_polarization, d_polarization, h_polarization_dim*sizeof(double), cudaMemcpyDeviceToHost));

   printf("Tempo d'esecuzione su GPU (Global Memory): %f ms.\n", milliseconds);

}


extern "C"
void launchTextureKernel (double* h_polarization, int* h_neighbours, int h_polarization_dim, int h_neighbours_dim) {

   // Declarations
   float milliseconds;
   cudaEvent_t startEvent, stopEvent;
   double* d_polarization;
   int* d_neighbours;	

   // Setup execution parameters
   dim3 threads(BLOCK_DIM);
   dim3 grid(ceil(h_polarization_dim/BLOCK_DIM));

   cudaSetDevice( cutGetMaxGflopsDeviceId() );

   // allocate device memory for result
   cutilSafeCall(cudaMalloc((void**) &d_polarization, h_polarization_dim*sizeof(double))); 
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
   textureKernel<<< grid, threads >>> (d_polarization, d_neighbours, h_polarization_dim, h_neighbours_dim);

   // Wait Device
   cudaThreadSynchronize();

   cutilSafeCall(cudaEventRecord(stopEvent, 0));  
   cutilSafeCall(cudaEventSynchronize(stopEvent));
   cutilSafeCall(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
   
   printf("Fine kernel\n");

   cutilSafeCall(cudaMemcpy(h_polarization, d_polarization, h_polarization_dim*sizeof(double), cudaMemcpyDeviceToHost));

   printf("Tempo d'esecuzione su GPU (Texture Memory): %f ms.\n", milliseconds);


}


