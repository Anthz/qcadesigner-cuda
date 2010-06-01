/**
TODO:
   1. double/Double
   2. Il problema delle celle Fixed/Input è risolto settando tutti i vicini a -1
   3. Parametri generate_next_clock. Valutare possibilità di generare next_clock nel kernel.
   4. Valutare la possibilità di rendere le dimensioni degli array e delle matrici multipli di BLOCK_DIM in modo da eliminare gli "if" nel kernel.
   5. Nel caso in cui double sia sufficiente, ottimizzare letture e scritture con double3
   6. Le define sparse per il codice sono state copiate anzicchè includere gli header... non è il massimo.
*/


#include <cutil_inline.h>
#include <cuda.h>
//#include "cuPrintf.cu"

extern "C"{
#include "design.h"
#include "objects/QCADCell.h"
#include "exp_array.h"
#include "coherence_vector.h"
}

#include <math.h>
#include <stdlib.h>
#include <string.h>

#undef	CLAMP
#define	CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define	BLOCK_DIM 64
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

// Debug-related defines
//#define DEBUG_ON

// Coherence Optimization
__constant__ double optimization_options_clock_prefactor;
__constant__ double optimization_options_clock_shift;
__constant__ double optimization_options_four_pi_over_number_samples;
__constant__ double optimization_options_two_pi_over_number_samples;
__constant__ double optimization_options_hbar_over_kBT;

// Coherence Options
__constant__ double options_clock_low;
__constant__ double options_clock_high;
__constant__ double options_clock_shift;
__constant__ double options_relaxation;
__constant__ double options_time_step;
__constant__ int options_algorithm;

// Other constants
__constant__ double clock_total_shift;
// TODO
// __constant__ double d_clock ???

__device__ inline double slope_x (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
   double mag = magnitude_energy_vector (PEk, Gamma);
   return (-(2.0 * Gamma * over_hbar / mag * tanh (optimization_options_hbar_over_kBT * mag) + lambda_x) / options_relaxation + (PEk * lambda_y * over_hbar));
}


__device__ inline double slope_y (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
   return -(options_relaxation * (PEk * lambda_x + 2.0 * Gamma * lambda_z) + hbar * lambda_y) / (options_relaxation * hbar);
}


__device__ inline double slope_z (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
   double mag = magnitude_energy_vector (PEk, Gamma);
   return (PEk * tanh (optimization_options_hbar_over_kBT * mag) + mag * (2.0 * Gamma * options_relaxation * lambda_y - hbar * lambda_z)) / (options_relaxation * hbar * mag);
}


// Next value of lambda x with choice of options_algorithm
__device__ inline double eval_next_lambda_x (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
	// TODO Possible implementation: adopt register to register intermediate
	// value, removing k1, k2, k3, k4 - if there are enough registers on the 
	// tesla for all the possible running warps
   double k1 = options_time_step * slope_x (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   double k2, k3, k4;

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
__device__ inline double eval_next_lambda_y (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
   double k1 = options_time_step * slope_y (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   double k2, k3, k4;

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
__device__ inline double eval_next_lambda_z (double t, double PEk, double Gamma, double lambda_x, double lambda_y, double lambda_z)
{
   double k1 = options_time_step * slope_z (t, PEk, Gamma, lambda_x, lambda_y, lambda_z);
   double k2, k3, k4;

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


// OK!
__device__ inline double generate_clock_at_sample_s 
(
	double clock_prefactor,
	int total_number_of_inputs,
	unsigned long int sample, 
	double four_pi_over_number_samples,
	unsigned int clock_num,				
	double total_clock_shift, //sum of last two terms in original clock's formula
	double options_clock_low,
	double	options_clock_high
)
{
	return CLAMP 
	(
		clock_prefactor * 
		cos 
		(
			((double) (1 << total_number_of_inputs)) * 
			(double)sample * 
			four_pi_over_number_samples - 
			(double)PI * 
			(double)clock_num * 0.5
		) + 
		total_clock_shift,
		options_clock_low,
		options_clock_high
	);
}

// TODO check for further corrections and todos inside code
__global__ void kernelIterationParallel 
(
	double *d_polarization, 
	double *d_lambda_x, 
	double *d_lambda_y, 
	double *d_lambda_z, 
	double *d_Ek, 
	unsigned int *d_clock,
	int *d_neighbours, 
	int cells_number, 
	int neighbours_number, 
	int sample_number, 
	int total_number_of_inputs
)
{

   int th_index; 
   int nb_index;   // Neighbour index
   int i;
   double clock_value;
   double PEk;
   double lambda_x, next_lambda_x;
   double lambda_y, next_lambda_y;
   double lambda_z, next_lambda_z;
   double t;

	th_index =  blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
	//cuPrintf("th_index=%d", th_index);

   // Only useful threads must work
   if (th_index < cells_number)
   {
      PEk = 0;
      for (i = 0; i < neighbours_number; i++)
      {
			if (d_neighbours[th_index*neighbours_number+i] != -1)
			{
	 			nb_index = d_neighbours[th_index*neighbours_number+i];
	 			PEk += d_polarization[nb_index] * d_Ek[th_index*neighbours_number+i];
			//	cuPrintf("nb_index: %d\td_polarization[nb_index]: %g\td_Ek[th_index*neighbours_number+i]: %g\n", nb_index, d_polarization[nb_index], d_Ek[th_index*neighbours_number+i]);
	 			// TODO Hyp: d_EK of i > actual_numof_neighbours == 1?
			}
      }

		//cuPrintf("%g\t", PEk);

		// Generate clock
      clock_value = 
		   generate_clock_at_sample_s 
			(
				optimization_options_clock_prefactor, 
				total_number_of_inputs,
				sample_number,
				optimization_options_four_pi_over_number_samples,
				d_clock[th_index],
				clock_total_shift,
				options_clock_low,
				options_clock_high
			);


		// TODO Optimization: remove these three doubles and use them directly into 
		// subsequent calls
      lambda_x = d_lambda_x[th_index];
      lambda_y = d_lambda_y[th_index];
      lambda_z = d_lambda_z[th_index];

      t = options_time_step * sample_number;
      next_lambda_x = eval_next_lambda_x (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);
      next_lambda_y = eval_next_lambda_y (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);
      next_lambda_z = eval_next_lambda_z (t, PEk, clock_value, lambda_x, lambda_y, lambda_z);

      d_lambda_x[th_index] = next_lambda_x;
      d_lambda_y[th_index] = next_lambda_y;
      d_lambda_z[th_index] = next_lambda_z;
      
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
void launch_coherence_vector_simulation (DESIGN *design, simulation_data *sim_data, QCADCell ***sorted_cells, coherence_optimizations *optimization_options, const coherence_OP *options, int number_of_cell_layers, int *number_of_cells_in_layer, int num_samples, int record_interval, int *STOP)
{
   // Variables
   double *h_polarization, *h_Ek, *h_lambda_x, *h_lambda_y, *h_lambda_z;
   unsigned int *h_clock;
   int *h_neighbours;

   double *d_polarization, *d_Ek, *d_lambda_x, *d_lambda_y, *d_lambda_z;
   unsigned int *d_clock;
   int *d_neighbours;

	FILE *fp;
   int i, j, k, h;
   int cells_number;
   int max_neighbours_number;
   int index;
	BUS_LAYOUT_ITER bli ;
  	double dPolarization = 2.0 ;
  	int idxMasterBitOrder = -1.0 ;
   double total_clock_shift = (optimization_options->clock_shift) + options->clock_shift;

    // Compute the number of cells, the max neighbours count and set the cuda_id field of each cell
   cells_number = 0;
   max_neighbours_number = 0;
   index = 0;
   for (i = 0; i < number_of_cell_layers; i++)
   {
      for (j = 0; j < number_of_cells_in_layer[i]; j++)
      {
	 		if (((coherence_model *)sorted_cells[i][j]->cell_model)->number_of_neighbours > max_neighbours_number)
	    		max_neighbours_number = ((coherence_model *)sorted_cells[i][j]->cell_model)->number_of_neighbours;
   
         sorted_cells[i][j]->cuda_id = index;

         index++;
      }
      cells_number += number_of_cells_in_layer[i];
   }

  // Set GPU Parameters
   dim3 threads (BLOCK_DIM);
   dim3 grid (ceil ((double)cells_number/BLOCK_DIM));

   // Set Devices
   cudaSetDevice (cutGetMaxGflopsDeviceId());
  // cudaPrintfInit ();

   // Allocate CUDA-Compatible structures
   h_polarization =	(double*) malloc (sizeof(double)*cells_number);
   h_clock =			(unsigned int*) malloc (sizeof(unsigned int)*cells_number);
   h_lambda_x =		(double*) malloc (sizeof(double)*cells_number);
   h_lambda_y =		(double*) malloc (sizeof(double)*cells_number);
   h_lambda_z =		(double*) malloc (sizeof(double)*cells_number);
   h_Ek =				(double*) malloc (sizeof(double)*cells_number*max_neighbours_number);
   h_neighbours =	(int*) malloc (sizeof(int)*cells_number*max_neighbours_number);

   // Initialize Memory
   cutilSafeCall (cudaMalloc (&d_polarization, cells_number*sizeof(double)));
   cutilSafeCall (cudaMalloc (&d_clock, cells_number*sizeof(unsigned int)));
   cutilSafeCall (cudaMalloc (&d_lambda_x, cells_number*sizeof(double)));
   cutilSafeCall (cudaMalloc (&d_lambda_y, cells_number*sizeof(double)));
   cutilSafeCall (cudaMalloc (&d_lambda_z, cells_number*sizeof(double)));
   cutilSafeCall (cudaMalloc (&d_Ek, sizeof(double)*max_neighbours_number*cells_number));
   cutilSafeCall (cudaMalloc (&d_neighbours, sizeof(int)*max_neighbours_number*cells_number));

   // Set Constants
   cutilSafeCall (cudaMemcpy (d_clock, h_clock, cells_number*sizeof(unsigned int), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_neighbours, h_neighbours, sizeof(int)*max_neighbours_number*cells_number, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_Ek, h_Ek, sizeof(double)*max_neighbours_number*cells_number, cudaMemcpyHostToDevice));

   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_prefactor", &(optimization_options->clock_prefactor), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_clock_shift", &(optimization_options->clock_shift), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_four_pi_over_number_samples", &(optimization_options->four_pi_over_number_samples), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_two_pi_over_number_samples", &(optimization_options->two_pi_over_number_samples), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("optimization_options_hbar_over_kBT", &(optimization_options->hbar_over_kBT), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_low", &(options->clock_low), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_high", &(options->clock_high), sizeof(double), 0, cudaMemcpyHostToDevice));  
   cutilSafeCall (cudaMemcpyToSymbol("options_clock_shift", &(options->clock_shift), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_relaxation", &(options->relaxation), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_time_step", &(options->time_step), sizeof(double), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("options_algorithm", &(options->algorithm), sizeof(int), 0, cudaMemcpyHostToDevice));
   cutilSafeCall (cudaMemcpyToSymbol("clock_total_shift", &(total_clock_shift), sizeof(double), 0, cudaMemcpyHostToDevice));

	//
	index = 0;
	fp = fopen ("cuda/log_coherence/circuit_structure", "w");
	for (i = 0; i < number_of_cell_layers; i++)
   {
      for (j = 0; j < number_of_cells_in_layer[i]; j++)
      {
			h_clock[index] = (sorted_cells[i][j]->cell_options).clock;
			fprintf (fp, "Cell: %d\n\tNeighbours (Ek):\n", index);
			for (k = 0; k < max_neighbours_number; k++)
		   {
		   	if (k < ((coherence_model *)sorted_cells[i][j]->cell_model)->number_of_neighbours)
		      {
		         h_Ek[index*max_neighbours_number+k] = ((coherence_model *)sorted_cells[i][j]->cell_model)->Ek[k];
		         h_neighbours[index*max_neighbours_number+k] = (((coherence_model *)sorted_cells[i][j]->cell_model)->neighbours[k])->cuda_id;
		      }
		      else
		      {
		         h_Ek[index*max_neighbours_number+k] = -1;
		         h_neighbours[index*max_neighbours_number+k] = -1;
		      }
				fprintf (fp, "\t\t%d(%g)\n", h_neighbours[index*max_neighbours_number+k], h_Ek[index*max_neighbours_number+k]);
		   }
			index++;
			fprintf (fp, "\n");
		}	
	}
	fclose (fp);

   // For each sample...
   for (j = 0; j < num_samples; j++)
   {
		for
		(
			idxMasterBitOrder = 0, design_bus_layout_iter_first (design->bus_layout, &bli, QCAD_CELL_INPUT, &i);
			i > -1 ;
			design_bus_layout_iter_next (&bli, &i), idxMasterBitOrder++
		)
		{
			qcad_cell_set_polarization (exp_array_index_1d (design->bus_layout->inputs, BUS_LAYOUT_CELL, i).cell,
          dPolarization = (-sin (((double) (1 << idxMasterBitOrder)) * (double) j * optimization_options->four_pi_over_number_samples)) > 0 ? 1 : -1) ;
			if (0 == j % record_interval)
				sim_data->trace[i].data[j/record_interval] = dPolarization ;
		}

		if (0 == j % record_interval)
		{
			for 
			(			
				design_bus_layout_iter_first ( design->bus_layout, &bli, QCAD_CELL_INPUT, &i ) ; 
				i > -1 ;
				design_bus_layout_iter_next ( &bli, &i)
			)
			{
				sim_data->trace[i].data[j/record_interval] = 
				qcad_cell_calculate_polarization (exp_array_index_1d (design->bus_layout->inputs, BUS_LAYOUT_CELL, i).cell);
			}
		}

		// Fill CUDA-Compatible structures
		index = 0;
		for (k = 0; k < number_of_cell_layers; k++)
		{
		   for (h = 0; h < number_of_cells_in_layer[k]; h++)
		   {
		      h_polarization[index] = qcad_cell_calculate_polarization(sorted_cells[k][h]);
		      h_lambda_x[index] = ((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_x;
		      h_lambda_y[index] = ((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_y;
		      h_lambda_z[index] = ((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_z;
		     
		      index++;
		   }
		}

		// Set Memory
		cutilSafeCall (cudaMemcpy (d_polarization, h_polarization, cells_number*sizeof(double), cudaMemcpyHostToDevice));
		cutilSafeCall (cudaMemcpy (d_lambda_x, h_lambda_x, cells_number*sizeof(double), cudaMemcpyHostToDevice));
		cutilSafeCall (cudaMemcpy (d_lambda_y, h_lambda_y, cells_number*sizeof(double), cudaMemcpyHostToDevice));
		cutilSafeCall (cudaMemcpy (d_lambda_z, h_lambda_z, cells_number*sizeof(double), cudaMemcpyHostToDevice));
				
      // Launch Kernel
      printf ("Iteration# %d...", j); 
      kernelIterationParallel<<< grid, threads >>> (d_polarization, d_lambda_x, d_lambda_y, d_lambda_z, d_Ek, d_clock, d_neighbours, cells_number, max_neighbours_number, j, design->bus_layout->inputs->icUsed);

      // Wait Device
      cudaThreadSynchronize ();

    //  cudaPrintfDisplay(stdout, true);

      // Return to Host lambdas values
      cutilSafeCall (cudaMemcpy (h_lambda_x, d_lambda_x, cells_number*sizeof(double), cudaMemcpyDeviceToHost));
      cutilSafeCall (cudaMemcpy (h_lambda_y, d_lambda_y, cells_number*sizeof(double), cudaMemcpyDeviceToHost));
      cutilSafeCall (cudaMemcpy (h_lambda_z, d_lambda_z, cells_number*sizeof(double), cudaMemcpyDeviceToHost));
		
		index = 0;
		for (k = 0; k < number_of_cell_layers; k++)
			for (h = 0; h < number_of_cells_in_layer[k]; h++)
			{
				// don't simulate the input and fixed cells //
				if 
				(
					(QCAD_CELL_INPUT == sorted_cells[k][h]->cell_function) ||
					(QCAD_CELL_FIXED == sorted_cells[k][h]->cell_function)
					
				)
				{
					index++;
					continue;
				}		
				// if polarization went mad, abort
				if 
				(
					fabs (((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_z) > 1.0
				)
				{
					//abort
					return ;
				}
				
				// if everything goes well, update polarizations for next cycle
				qcad_cell_set_polarization
				(
					//sorted_cells[k][h], ((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_z
					sorted_cells[k][h], h_lambda_z[index]
				);
				
				((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_x = h_lambda_x[index];
				((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_y = h_lambda_y[index];
				((coherence_model *)sorted_cells[k][h]->cell_model)->lambda_z = h_lambda_z[index];

				index++;
			}

			printf("Complete!\n");

			char str[256] = "cuda/log_coherence/";
			char num[10];
		
			sprintf (num, "%i", j);
			strcat (str, num);

			fp = fopen(str, "w");
			for( k = 0; k < cells_number; k++)
			{
				fprintf(fp,"cell %d: %f\n", k, h_lambda_z[k]);
			}

			fclose (fp);
			
		   // collect all the output data from the simulation
			if (0 == j % record_interval)
			{
				for 
				(			
					design_bus_layout_iter_first (design->bus_layout, &bli, QCAD_CELL_OUTPUT, &i); 
					i > -1 ;
					design_bus_layout_iter_next (&bli, &i)
				)
				{
					sim_data->trace[(design->bus_layout->inputs->icUsed)+i].data[j/record_interval] =
					qcad_cell_calculate_polarization (exp_array_index_1d (design->bus_layout->outputs, BUS_LAYOUT_CELL, i).cell);
				}
			}

		if (TRUE == *STOP) 
		{
			// Free-up resources
//			cudaPrintfEnd();
			cudaFree(d_polarization);
			cudaFree(d_clock);
			cudaFree(d_lambda_x);
			cudaFree(d_lambda_y);
			cudaFree(d_lambda_z);
			cudaFree(d_Ek);
			cudaFree(d_neighbours);    

			return;
		}
   }

   // Free-up resources
  // cudaPrintfEnd();
   cudaFree(d_polarization);
   cudaFree(d_clock);
   cudaFree(d_lambda_x);
	cudaFree(d_lambda_y);
	cudaFree(d_lambda_z);
   cudaFree(d_Ek);
   cudaFree(d_neighbours);  

	return;
}




