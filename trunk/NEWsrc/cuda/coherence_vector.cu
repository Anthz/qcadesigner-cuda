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
#define magnitude_energy_vector(P,G) (hypot(2*(G), (P)) * over_hbar) /* (sqrt((4.0*(G)*(G) + (P)*(P))*over_hbar_sqr)) */

// Physical Constants
#define hbar 1.05457266e-34
#define over_hbar 9.48252e33
#define hbar_sqr 1.11212e-68
#define over_hbar_sqr 8.99183e67
#define kB 1.381e-23
#define over_kB 7.24296e22
#define E 1.602e-19


// Coherence Optimization
__constant__ float optimization_options_clock_prefactor;
__constant__ float optimization_options_clock_shift;
__constant__ float optimization_options_four_pi_over_number_samples;
__constant__ float optimization_options_two_pi_over_number_samples;
__constant__ float optimization_options_hbar_over_kBT;

// Coherence Options
__constant__ float options_relaxation;
__constant__ float options_time_step;
__constant__ int options_algorithm;


__device__ inline float slope_x (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float mag = magnitude_energy_vector (PEk, Gamma);
   return (-(2.0 * Gamma * over_hbar / mag * tanh (optimization_options_hbar_over_kBT * mag) + lambda_x) / options_relaxation + (PEk * lambda_y * over_hbar));
}


__device__ inline float slope_y (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   return -(options_relaxation * (PEk * lambda_x + 2.0 * Gamma * lambda_z) + hbar * lambda_y) / (options_relaxation * hbar);
}


__device__ inline float slope_z (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float mag = magnitude_energy_vector (PEk, Gamma);
   return (PEk * tanh (optimization_options_hbar_over_kBT * mag) + mag * (2.0 * Gamma * options_relaxation * lambda_y - hbar * lambda_z)) / (options_relaxation * hbar * mag);
}

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



// Next value of lambda x with choice of options_algorithm
__device__ inline float lambda_x_next (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_x (t, PEk, Gamma, lambda_x, lambda_y, lambda_z, options_relaxation);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k1/2, lambda_y, lambda_z, options_relaxation);
      k3 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k2/2, lambda_y, lambda_z, options_relaxation);
      k4 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k3,   lambda_y, lambda_z, options_relaxation);
      return lambda_x + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_x + k1;
   else
      return 0;
}


// Next value of lambda y with choice of options_algorithm
__device__ inline float lambda_y_next (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y, lambda_z, options_relaxation);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k1/2, lambda_z, options_relaxation);
      k3 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k2/2, lambda_z, options_relaxation);
      k4 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k3,   lambda_z, options_relaxation);
      return lambda_y + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_y + k1;
   else
      return 0;
}


// Next value of lambda z with choice of options_algorithm
__device__ inline float lambda_z_next (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_z (t, PEk, Gamma, lambda_x, lambda_y, lambda_z, options_relaxation);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k1/2, options_relaxation);
      k3 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k2/2, options_relaxation);
      k4 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k3,   options_relaxation);
      return lambda_z + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_z + k1;
   else
      return 0;
}


__global__ void kernel (float* d_next_polarization, float *d_polarization, float *d_clock, float *d_lambda_x, float *d_lambda_y, float *d_lambda_z, float **d_Ek, int **d_neighbours, int cells_number, int neighbours_number, int sample_number)
{

   int th_index = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
   int nb_index;   // Neighbour index
   int i;
   float PEk;
   float lambda_x, next_lambda_x;
   float lambda_y, next_lambda_y;
   float lambda_z, next_lambda_z;
   float t;

   // Only usefull threads must work
   if (th_index < cells_number)
   {
      t = options_time_step * sample_number;

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

      next_lambda_x = eval_next_lambda_x (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);
      next_lambda_y = eval_next_lambda_y (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);
      next_lambda_z = eval_next_lambda_z (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);

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
void launch_coherence_vector_simulation (float *h_polarization, float *h_clock, float *h_lambda_x, float *h_lambda_y, float *h_lambda_z, float **h_Ek, int **h_neighbours, int cells_number, int neighbours_number, int iterations, coherence_OP *options, coherence_optimizations *optimization_options)
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

   // Set Constants
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_prefactor", optimization_options->clock_prefactor, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_shift", optimization_options->clock_shift, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_four_pi_over_number_samples", optimization_options->four_pi_over_number_samples, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_two_pi_over_number_samples", optimization_options->two_pi_over_number_samples, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_hbar_over_kBT", optimization_options->hbar_over_kBT, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_relaxation", options->relaxation, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_time_step", options->time_step, sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_algorithm", options->algorithm, sizeof(int), 0, cudaMemcpyHostToDevice));

   // For each sample...
   for (i = 0; i < iterations; i++)
   {
      // Launch Kernel
      kernel<<< grid, threads >>> (d_next_polarization, d_polarization, d_clock, d_lambda_x, d_lambda_y, d_lambda_z, d_Ek, d_neighbours, cells_number, neighbours_number, i);

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


