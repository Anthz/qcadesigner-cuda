/**
TODO:
   1. Float/Double
   2. Il problema delle celle Fixed/Input è risolto settando tutti i vicini a -1
   3. Parametri generate_next_clock. Valutare possibilità di generare next_clock nel kernel.
   4. Valutare la possibilità di rendere le dimensioni degli array e delle matrici multipli di BLOCK_DIM in modo da eliminare gli "if" nel kernel.
   5. Nel casdo in cui Float sia sufficiente, ottimizzare letture e scritture con Float3
   6. Le define sparse per il codice sono state copiate anzicchè includere gli header... non è il massimo.
*/


#include <cutil_inline.h>
#include <cuda.h>
#include "cuPrintf.cu"
#include "structs.h"

#include <math.h>

#undef	CLAMP
#define	CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define	BLOCK_DIM 256
#define	magnitude_energy_vector(P,G) (hypot(2*(G), (P)) * over_hbar) /* (sqrt((4.0*(G)*(G) + (P)*(P))*over_hbar_sqr)) */

// Physical Constants (from coherence_vector.h)
#define hbar 1.05457266e-34
#define over_hbar 9.48252e33
#define hbar_sqr 1.11212e-68
#define over_hbar_sqr 8.99183e67
#define kB 1.381e-23
#define over_kB 7.24296e22
#define E 1.602e-19

// Simulation Types (from global_consts.h)
#define EXHAUSTIVE_VERIFICATION 0
#define VECTOR_TABLE 1

// Simulation Algorithms (from global_consts.h)
#define RUNGE_KUTTA 1
#define EULER_METHOD 2

// Some useful physical constants (from global_consts.h)
#define QCHARGE_SQUAR_OVER_FOUR 6.417423538e-39
#define QCHARGE 1.602176462e-19
#define HALF_QCHARGE 0.801088231e-19
#define OVER_QCHARGE 6.241509745e18
#define ONE_OVER_FOUR_HALF_QCHARGE 3.12109e18
#define EPSILON 8.8541878e-12
#define PI 3.1415926535897932384626433832795
#define FOUR_PI 12.56637061
#define FOUR_PI_EPSILON 1.112650056e-10
#define HBAR 1.0545887e-34
#define PRECISION 1e-5


// Coherence Optimization
__constant__ float optimization_options_clock_prefactor;
__constant__ float optimization_options_clock_shift;
__constant__ float optimization_options_four_pi_over_number_samples;
__constant__ float optimization_options_two_pi_over_number_samples;
__constant__ float optimization_options_hbar_over_kBT;

// Coherence Options
__constant__ float options_clock_low;
__constant__ float options_clock_high;
__constant__ float options_clock_shift;
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


// Next value of lambda x with choice of options_algorithm
__device__ inline float eval_next_lambda_x (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_x (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k1/2, lambda_y, lambda_z);
      k3 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k2/2, lambda_y, lambda_z);
      k4 = options_time_step * slope_x (t, PEk, Gamma, lambda_x + k3,   lambda_y, lambda_z);
      return lambda_x + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_x + k1;
   else
      return 0;
}


// Next value of lambda y with choice of options_algorithm
__device__ inline float eval_next_lambda_y (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k1/2, lambda_z);
      k3 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k2/2, lambda_z);
      k4 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y + k3,   lambda_z);
      return lambda_y + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_y + k1;
   else
      return 0;
}


// Next value of lambda z with choice of options_algorithm
__device__ inline float eval_next_lambda_z (float t, float PEk, float Gamma, float lambda_x, float lambda_y, float lambda_z)
{
   float k1 = options_time_step * slope_z (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   float k2, k3, k4;

   if (RUNGE_KUTTA == options_algorithm)
   {
      k2 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k1/2);
      k3 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k2/2);
      k4 = options_time_step * slope_z(t, PEk, Gamma, lambda_x, lambda_y, lambda_z + k3  );
      return lambda_z + k1/6 + k2/3 + k3/3 + k4/6;
   }
   else
   if (EULER_METHOD == options_algorithm)
      return lambda_z + k1;
   else
      return 0;
}


__device__ inline float generate_clock_at_sample_s (unsigned int clock_num, unsigned long int sample, unsigned long int number_samples, int total_number_of_inputs, const CUDA_coherence_OP *options, int SIMULATION_TYPE /*, VectorTable *pvt*/)
{
/*
   float clock = optimization_options_clock_prefactor * cos (((float) (1 << total_number_of_inputs)) * (float) sample * optimization_options_four_pi_over_number_samples - PI * (float)clock_num * 0.5) + optimization_options_clock_shift + options_clock_shift;

   // Saturate the clock at the clock high and low values
	clock = CLAMP (clock, options_clock_low, options_clock_high) ;
	return clock;
*/
	return CLAMP (optimization_options_clock_prefactor * cos (((float) (1 << total_number_of_inputs)) * (float) sample * optimization_options_four_pi_over_number_samples - PI * (float)clock_num * 0.5) + optimization_options_clock_shift + options_clock_shift, options_clock_low, options_clock_high);
}

/*
__global__ void kernelIterationParallelOptimized (float* d_next_polarization, float *d_polarization, float *d_lambda_x, float *d_lambda_y, float *d_lambda_z, float *d_Ek, int *d_neighbours, int cells_number, int neighbours_number, int sample_number, int number_of_inputs)
{

   int th_index = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
   int nb_index;   // Neighbour index
   int i;
   float clock_value;
   float PEk;
   float lambda_x, next_lambda_x;
   float lambda_y, next_lambda_y;
   float lambda_z, next_lambda_z;
   float t;
   int total_number_of_inputs = number_of_inputs;

   // Only usefull threads must work
   if (th_index < cells_number)
   {
      t = options_time_step * sample_number;

      // Generate clock
		clock_value = CLAMP (optimization_options_clock_prefactor * cos (((float) (1 << total_number_of_inputs)) * (float) sample_number * optimization_options_four_pi_over_number_samples - PI * (float)clock_num * 0.5) + optimization_options_clock_shift + options_clock_shift, options_clock_low, options_clock_high);

      PEk = 0;
   
      for (i = 0; i < neighbours_number; i++)
      {
	 nb_index = d_neighbours[th_index*neighbours_number+i];
	 PEk += d_polarization[nb_index] * d_Ek[th_index*neighbours_number+nb_index]; 
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

      cuPrintf("polarization: %f\tclock: %f\tlambda: %f %f %f\tEk: %f\n", d_polarization[th_index], clock_value, d_lambda_x[th_index], d_lambda_y[th_index], d_lambda_z[th_index], d_Ek[th_index]);
   }

}
*/

__global__ void kernelIterationParallel (float* d_next_polarization, float *d_polarization, float *d_lambda_x, float *d_lambda_y, float *d_lambda_z, float *d_Ek, int *d_neighbours, int cells_number, int neighbours_number, int sample_number, int number_of_inputs)
{

   int th_index = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
   int nb_index;   // Neighbour index
   int i;
   float clock_value;
   float PEk;
   float lambda_x, next_lambda_x;
   float lambda_y, next_lambda_y;
   float lambda_z, next_lambda_z;
   float t;
   int total_number_of_inputs = number_of_inputs;

   // Only usefull threads must work
   if (th_index < cells_number)
   {
      t = options_time_step * sample_number;

      // Generate clock
		clock_value = 5; //generate_clock_at_sample_s (h_clock, cells_number, i, ...)

      PEk = 0;
   
      for (i = 0; i < neighbours_number; i++)
      {
	 nb_index = d_neighbours[th_index*neighbours_number+i];
	 PEk += d_polarization[nb_index] * d_Ek[th_index*neighbours_number+nb_index]; 
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

      cuPrintf("polarization: %f\tclock: %f\tlambda: %f %f %f\tEk: %f\n", d_polarization[th_index], clock_value, d_lambda_x[th_index], d_lambda_y[th_index], d_lambda_z[th_index], d_Ek[th_index]);
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
void launch_coherence_vector_simulation (float *h_polarization, float *h_lambda_x, float *h_lambda_y, float *h_lambda_z, float *h_Ek, int *h_neighbours, int cells_number, int neighbours_number, int iterations, CUDA_coherence_OP *options, CUDA_coherence_optimizations *optimization_options)
{

   // Variables
   float *d_next_polarization, *d_polarization, *d_Ek, *d_lambda_x, *d_lambda_y, *d_lambda_z;
   int *d_neighbours;
   int i;

   // Set GPU Parameters
   dim3 threads (BLOCK_DIM);
   dim3 grid (ceil ((float)cells_number/BLOCK_DIM));

   // Set Devices
   cudaSetDevice (cutGetMaxGflopsDeviceId());
   cudaPrintfInit ();

   // Initialize Memory
   cutilSafeCall (cudaMalloc (&d_next_polarization, cells_number*sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_polarization, cells_number*sizeof(float))); 
   cutilSafeCall (cudaMalloc (&d_lambda_x, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_lambda_y, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_lambda_z, cells_number*sizeof(float)));
   cutilSafeCall (cudaMalloc (&d_Ek, sizeof(float)*neighbours_number*cells_number));
   cutilSafeCall (cudaMalloc (&d_neighbours, sizeof(int)*neighbours_number*cells_number));

   // Set Memory
   cutilSafeCall (cudaMemcpy (d_polarization, h_polarization, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_x, h_lambda_x, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_y, h_lambda_y, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_lambda_z, h_lambda_z, cells_number*sizeof(float), cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_Ek, h_Ek, sizeof(float)*neighbours_number*cells_number, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpy (d_neighbours, h_neighbours, sizeof(int)*neighbours_number*cells_number, cudaMemcpyHostToDevice));

   // Set Constants
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_prefactor", &(optimization_options->clock_prefactor), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_shift", &(optimization_options->clock_shift), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_four_pi_over_number_samples", &(optimization_options->four_pi_over_number_samples), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_two_pi_over_number_samples", &(optimization_options->two_pi_over_number_samples), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_hbar_over_kBT", &(optimization_options->hbar_over_kBT), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_low", &(options->clock_low), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_high", &(options->clock_high), sizeof(float), 0, cudaMemcpyHostToDevice));  
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_shift", &(options->clock_shift), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_relaxation", &(options->relaxation), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_time_step", &(options->time_step), sizeof(float), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_algorithm", &(options->algorithm), sizeof(int), 0, cudaMemcpyHostToDevice));

   // For each sample...
   for (i = 0; i < iterations; i++)
   {
      printf("Inizio simulazione -- Iterazione %d\n", i); 

      // Launch Kernel
      kernelIterationParallel<<< grid, threads >>> (d_next_polarization, d_polarization, d_lambda_x, d_lambda_y, d_lambda_z, d_Ek, d_neighbours, cells_number, neighbours_number, i, 10);

      // Wait Device
      cudaThreadSynchronize ();

      cudaPrintfDisplay(stdout, true);

      // Set Memory for the next iteration
      cutilSafeCall (cudaMemcpy (d_polarization, d_next_polarization, cells_number*sizeof(float), cudaMemcpyDeviceToDevice));

      // Test -- Get desidered iteration results from GPU
      cutilSafeCall (cudaMemcpy (h_polarization, d_next_polarization, cells_number*sizeof(float), cudaMemcpyDeviceToHost));
   }

   // Free-up resources
   cudaPrintfEnd();
   cudaFree(d_next_polarization);
   cudaFree(d_polarization);
   cudaFree(d_Ek);
   cudaFree(d_neighbours);  

}



