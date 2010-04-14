/**
TODO:
   1. Float/Double
   2. Il problema delle celle Fixed/Input è risolto settando tutti i vicini a -1
   3. Parametri generate_next_clock. Valutare possibilità di generare next_clock nel kernel.
   4. Valutare la possibilità di rendere le dimensioni degli array e delle matrici multipli di BLOCK_DIM in modo da eliminare gli "if" nel kernel.
   5. Nel casdo in cui Float sia sufficiente, ottimizzare letture e scritture con Float3
*/


#include <cutil_inline.h>
#include <cuda.h>

#include <math.h>

#undef	CLAMP
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define BLOCK_DIM 256

__device__ inline double generate_clock_at_sample_s (unsigned int clock_num, unsigned long int sample, unsigned long int number_samples, int total_number_of_inputs, const coherence_OP *options, int SIMULATION_TYPE, VectorTable *pvt)
{
  double clock = 0;

  /*
   if (SIMULATION_TYPE == EXHAUSTIVE_VERIFICATION)
   {
  */
    clock = clock_prefactor *
      cos (((double) (1 << total_number_of_inputs)) * (double) sample * optimization_options_four_pi_over_number_samples - PI * (double)clock_num * 0.5) + optimization_options_clock_shift + options_clock_shift;

    // Saturate the clock at the clock high and low values
    clock = CLAMP (clock, options_clock_low, options_clock_high) ;
  /*
  }
  else
  if (SIMULATION_TYPE == VECTOR_TABLE)
    {
    clock = clock_prefactor *
      cos (((double)pvt->vectors->icUsed) * (double) sample * optimization_options.two_pi_over_number_samples - PI * (double)clock_num * 0.5) + optimization_options.clock_shift + options->clock_shift;

    // Saturate the clock at the clock high and low values
    clock = CLAMP (clock, options->clock_low, options->clock_high) ;
    }
   */
  return clock;
  }// calculate_clock_value



__global__ void kernel (float* d_next_polarization, float *d_polarization, float *d_clock, float *d_lambda_x, float *d_lambda_y, float *d_lambda_z, float **d_Ek, int **d_neighbours, int cells_number, int neighbours_number)
{

   int th_index = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
   int nb_index;   // Neighbour index
   int i;
   float PEk;
   float lambda_x, next_lambda_x;
   float lambda_y, next_lambda_y;
   float lambda_z, next_lambda_z;

   // Only usefull threads must work
   if (th_index < cells_number)
   {
      // Generate next clock
      generate_clock_at_sample_s (h_clock, cells_number, i, ...);

      PEk = 0;
      
      for (i = 0; i < neighbours_number; i++)
      {
	 nb_index = d_neighbours[th_index][i];
	 PEk += d_polarization[nb_index] * d_Ek[th_index][nb_index]; 
      }

      lambda_x = d_lambda_x[th_index];
      lambda_y = d_lambda_y[th_index];
      lambda_z = d_lambda_z[th_index];

      next_lambda_x = eval_next_lambda_x (...);
      next_lambda_y = eval_next_lambda_y (...);
      next_lambda_z = eval_next_lambda_z (...);

      d_lambda_x[th_index] = next_lambda_x;
      d_lambda_y[th_index] = next_lambda_y;
      d_lambda_z[th_index] = next_lambda_z;
      
      d_next_polarization[th_index] = next_lambda_z;
   }

}


/**
 \param <h_polarization> {Vector containing cells polarization (Host side).}
 \param <h_clock> {}
 \param <h_Ek> {}
 \param <h_neighbours> {}
 \param <cells_number> {}
 \param <neighbours_number> {}
 \param <iteration> {}
*/
extern "C"
void launch_coherence_vector_simulation (float *h_polarization, float *h_clock, float *h_lambda_x, float *h_lambda_y, float *h_lambda_z, float **h_Ek, int **h_neighbours, int cells_number, int neighbours_number, int iterations)
{

   // Variables
   size_t Ek_pitch, neighbours_pitch;
   float *d_next_polarization, *d_polarization, *d_clock, **d_Ek;
   int **d_neighbours;
   int i;

   // Set GPU Parameters
   dim3 threads (BLOCK_DIM);
   dim3 grid (ceil ((float)h_polarization_dim/BLOCK_DIM));

   // Set Devices
   cudaSetDevice (cutGetMaxGflopsDeviceId());
   cudaPrintfInit ();

   // Initialize Memory
   cutilSafeCall (cudaMalloc (&d_next_polarization, cells_number*sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_polarization, cells_number*sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_clock, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_lambda_x, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_lambda_y, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_lambda_z, cells_number*sizeof(float)));
   cutilSafeCall (cudaMallocPitch (&d_Ek, &Ek_pitch, sizeof(float)*neighbours_number, cells_number));
   cutilSafeCall (cudaMallocPitch (&d_neighbours, &neighbours_pitch, sizeof(int)*neighbours_number, cells_number));

   // Set Memory
   cutilSafeCall (cudaMemcpy (d_polarization, h_polarization, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_clock, h_clock, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_x, h_lambda_x, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_y, h_lambda_y, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_z, h_lambda_z, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy2D (d_Ek, Ek_pitch, h_Ek, 0, sizeof(float)*neighbours_number, cells_number, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy2D (d_neighbours, neighbours_pitch, h_neighbours, 0, sizeof(int)*neighbours_number, cells_number, cudaMemcpyHostToDevice));

   // For each sample...
   for (i = 0; i < iterations; i++)
   {
      // Launch Kernel
      kernel<<< grid, threads >>> (d_next_polarization, d_polarization, d_clock, d_lambda_x, d_lambda_y, d_lambda_z, d_Ek, d_neighbours, cells_number, neighbours_number);

      // Wait Device
      cudaThreadSynchronize ();

      // Set Memory for the next iteration
      cutilSafeCall (cudaMemcpy (d_polarization, d_next_polarization, cells_number*sizeof(float), cudaMemcpyDeviceToDevice));

      // Get desidered iteration results from GPU
      ...
   }

   // Free-up resources
   cudaPrintfEnd();
   cudaFree(d_next_polarization);
   cudaFree(d_polarization);
   cudaFree(d_clock);
   cudaFree(d_Ek);
   cudaFree(d_neighbours);  

}


