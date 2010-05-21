/* ========================================================================== */
/*                                                                            */
/*  CUDA_bistable_iteration.cu                                                */
/*    0- controllare che non vi siano scritte delle porcate da me medesimo*/
/*  1- valutare possibilità di unrollare il loop sui neighbours               */
/*  (visto che ne stabiliamo il numero di iterazioni a priori)                */
/*  2- il controllo sulle celle fixed crea una bella divergenza... proposte?  */
/*  3- 19maggio: clock_data troooppo grande					*/
/*  --> meglio farsi una memcpy ogni sample di clock_data[4] e d_polarization
	con i nuovi valori di polarizzazione degli input (ancora DA MODIFICARE!)*/

/* ========================================================================== */


#include <cutil_inline.h>
#include <cuda.h>
//#include "cuPrintf.cu"


#include <math.h>

#define BLOCK_DIM 256

  __global__ void bistable_kernel (float* d_polarization, float *d_next_polarization, int *d_cell_clock, float *d_clock_data, float *d_Ek, int *d_neighbours)
  {

   int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
   int nb_idx;   // Neighbour index
   int q;
   int current_cell_clock;   //could be 0, 1, 2 or 3
   float new_polarization;
   float polarization_math;
   int cells_number, neighbours_number, sample;
  
   // Only useful threads must work
   if (thr_idx < cells_number)
   {

      if (!(d_neighbours[thr_idx * neighbours_number] == -1)) // if thr_idx corresponding cell type is FIXED or INPUT
      {
        polarization_math = 0;
/*        
	for(q = 0; q < neighbours_number; q++)
        {
         nb_idx = d_neighbours[thr_idx * neighbours_number + q];
         polarization_math += d_Ek[thr_idx * neighbours_number + q] * d_polarization[nb_idx];
        }
         */
         //math = math / 2 * gamma
         current_cell_clock  = d_cell_clock[thr_idx];
         polarization_math /= (2.0 * d_clock_data[sample*4 + current_cell_clock]); // ...abbozzo
         
         // -- calculate the new cell polarization -- //
         // if math < 0.05 then math/sqrt(1+math^2) ~= math with error <= 4e-5
         // if math > 100 then math/sqrt(1+math^2) ~= +-1 with error <= 5e-5
            new_polarization =
              (polarization_math        >  1000.0)   ?  1                 :
              (polarization_math        < -1000.0)   ? -1                 :
              (fabs (polarization_math) <     0.001) ?  polarization_math :
                polarization_math / sqrt (1 + polarization_math * polarization_math) ;
         
          //set the new polarization in next_polarization array  
            d_next_polarization[thr_idx] = new_polarization;
          
          
          // -->>> nel caso volessimo considerare la stabilità...
          // If any cells polarization has changed beyond this threshold
          // then the entire circuit is assumed to have not converged.          
          //  stable = (fabs (new_polarization - old_polarization) <= tolerance) ;                
          //  d_stability[thr_idx] = stable;
        }
      
	else 
         //for FIXED and INPUT type cells polarization remains the same
         d_next_polarization[thr_idx] = d_polarization[thr_idx]; 
      }
      
    }
   
extern "C"
void launch_bistable_simulation(float *h_polarization, float *h_Ek, int *h_cell_clock, float *h_clock_data, int *h_neighbours, int cells_number, int neighbours_number, int number_of_samples, int iterations_per_sample)
{

printf("\nentrato nella launch!\n");
 // Variables
   float *d_next_polarization, *d_polarization, *d_clock_data, *d_Ek;
   int *d_neighbours, *d_cell_clock;
   int i;

   // Set GPU Parameters
   dim3 threads (BLOCK_DIM);
   dim3 grid (ceil ((float)cells_number/BLOCK_DIM));

   // Set Devices
   cudaSetDevice (cutGetMaxGflopsDeviceId());
//   cudaPrintfInit ();

   // Initialize Memory
   cutilSafeCall (cudaMalloc (&d_next_polarization, cells_number * sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_polarization, cells_number * sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_cell_clock, cells_number * sizeof(int)));
   cutilSafeCall (cudaMalloc (&d_clock_data, 4 * number_of_samples * sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_Ek, sizeof(float)*neighbours_number*cells_number));
   cutilSafeCall (cudaMalloc (&d_neighbours, sizeof(int)*neighbours_number*cells_number));

printf("malloc eseguite\n");
   // Set Memory
   cutilSafeCall (cudaMemcpy (d_polarization, h_polarization, cells_number * sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_cell_clock, h_cell_clock, cells_number * sizeof(int), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_clock_data, h_clock_data, 4 * number_of_samples * sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_Ek, h_Ek, sizeof(float) * neighbours_number * cells_number, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_neighbours, h_neighbours, sizeof(int) * neighbours_number * cells_number, cudaMemcpyHostToDevice));
printf("memcpy eseguite\n");

int j;

 for (j = 0; j < number_of_samples ; j++)
  {


  // In each sample...
   for (i = 0; i < iterations_per_sample; i++) //we are not considering stability
   {
      // Launch Kernel
      bistable_kernel<<< grid, threads >>> (d_polarization, d_next_polarization, d_cell_clock, d_clock_data, d_Ek, d_neighbours);

      // Wait Device
    //  cudaThreadSynchronize ();

//printf("j:%d \t i=%d\n", j,i);      

      // Set Memory for the next iteration
   //   cutilSafeCall (cudaMemcpy (d_polarization, d_next_polarization, cells_number * sizeof(float), cudaMemcpyDeviceToDevice));
      
    }
	// Get desidered iteration results from GPU
 //  cutilSafeCall (cudaMemcpy (h_polarization, d_polarization, cells_number * sizeof(float), cudaMemcpyDeviceToHost));
  }
      
      printf("\nfacciamo la clean\n");      
// Free-up resources
//   cudaPrintfEnd();
   cudaFree(d_next_polarization);
   cudaFree(d_polarization);
   cudaFree(d_cell_clock);
   cudaFree(d_clock_data);
   cudaFree(d_Ek);
   cudaFree(d_neighbours);  




}
