/* ========================================================================== */
/*                                                                            */
/*  CUDA_bistable_iteration.cu                                                */
/*    0- controllare che non vi siano scritte delle porcate da me medesimo*/
/*  1- valutare possibilità di unrollare il loop sui neighbours               */
/*  (visto che ne stabiliamo il numero di iterazioni a priori)                */
/*  2- il controllo sulle celle fixed crea una bella divergenza... proposte?  */
/*  3- 19maggio: clock_data troooppo grande
/*  --> meglio farsi una memcpy ogni sample di clock_data[4] e d_polarization
	con i nuovi valori di polarizzazione degli input (ancora DA MODIFICARE!)
					*/
/* ========================================================================== */


#include <cutil_inline.h>
#include <cuda.h>
//#include "cuPrintf.cu"
#include <time.h>

#include <math.h>

#define BLOCK_DIM 256
#undef CLAMP
#define CLAMP(value,low,high) ((value > high) ? high : ((value < low) ? low : value))
#undef PI
#define PI  3.14159265358979323846

__constant__ double d_clock_prefactor;
__constant__ double d_clock_shift;
__constant__ int d_cells_number;
__constant__ int d_neighbours_number;
__constant__ int d_input_number;
__constant__ int d_output_number;
__constant__ int d_number_of_samples;
__constant__ double d_clock_low;
__constant__ double d_clock_high;

extern	__shared__ int shm_array[];

__device__ inline int find(int x, int *array, int length)
{
	int l = 0, r = length - 1, mid;
	while (l <= r)
	{
		mid = (l + r) / 2;
		if (array[mid] == x) return mid;
		else if (array[mid] > x) r = mid - 1;
		else l = mid + 1;
	}
	return -1;
}


__global__ void update_inputs (double *d_polarization, int *d_input_indexes, int sample)
{
	int input_idx;
	int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (threadIdx.x < d_input_number)
	{
		shm_array[threadIdx.x] = d_input_indexes[threadIdx.x];
	}
	
	__syncthreads();
	
	input_idx = find(thr_idx, shm_array, d_input_number);

	if (input_idx >= 0)
	{
		d_polarization[thr_idx] = (-1 * sin(( 1 << input_idx) * sample * 4.0 * PI / d_number_of_samples) > 0 ) ? 1.0 : -1.0;
	}
}


__global__ void bistable_kernel (
		double *d_polarization,
		double *d_next_polarization,
		int *d_cell_clock,
		double *d_Ek,
		int *d_neighbours,
		int sample,
		int *d_output_indexes,
		int *d_stability,
		double tolerance,
		double *d_output_data
		)
{
	int thr_idx = blockIdx.x * blockDim.x + threadIdx.x;   // Thread index
	int nb_idx;   // Neighbour index
	int q;
	int current_cell_clock;   //could be 0, 1, 2 or 3
	double new_polarization;
	double polarization_math;
	double clock_value;
	int input_idx;
	int output_idx;
	int stable;
	int *shm_output_indexes = shm_array;
	double nb_pol;
  
	if (threadIdx.x < d_output_number)
	{
		shm_output_indexes[threadIdx.x] = d_output_indexes[threadIdx.x];
	}

	__syncthreads();

	// Only useful threads must work
	if (thr_idx < d_cells_number)
	{		
		//cuPrintf("PRUGNE!!");
		//cuPrintf("\nd_output_number = %d,\t d_output_indexes[0]=%d\n",d_output_number, d_output_indexes[0] );	
		  
		//cuPrintf("%f ", d_polarization[thr_idx]);	

		if (!(d_neighbours[thr_idx] == -1)) // if thr_idx corresponding cell type is not FIXED or INPUT
		{
			polarization_math = 0;
			for(q = 0; q < d_neighbours_number; q++)
			{
				nb_idx = d_neighbours[thr_idx + q * d_cells_number];
				if (nb_idx != -1) polarization_math += d_Ek[thr_idx + q * d_cells_number] * d_polarization[nb_idx];
			}

			//math = math / 2 * gamma
			current_cell_clock  = d_cell_clock[thr_idx];
			clock_value = d_clock_prefactor * cos ((1 << d_input_number) * sample * 4.0 * PI / d_number_of_samples - PI * current_cell_clock / 2.0) + d_clock_shift;
			clock_value = CLAMP(clock_value,d_clock_low,d_clock_high);
			polarization_math /= (2.0 * clock_value);
			 
			// -- calculate the new cell polarization -- //
			// if math < 0.05 then math/sqrt(1+math^2) ~= math with error <= 4e-5
			// if math > 100 then math/sqrt(1+math^2) ~= +-1 with error <= 5e-5
			new_polarization =
			(polarization_math        >  1000.0)   ?  1.0                 :
			(polarization_math        < -1000.0)   ? -1.0                 :
			(fabs (polarization_math) <     0.001) ?  polarization_math :
			polarization_math / sqrt (1 + polarization_math * polarization_math) ;

			//set the new polarization in next_polarization array  
			d_next_polarization[thr_idx] = new_polarization;

				  

			// If any cells polarization has changed beyond this threshold
			// then the entire circuit is assumed to have not converged.      
			stable = (fabs (new_polarization - d_polarization[thr_idx]) <= tolerance);
			//d_stability[thr_idx] = 1;
			d_stability[thr_idx] = stable;
			//cuPrintf("\n new:%f, old:%f \n", new_polarization, d_polarization[thr_idx]);

			output_idx = find(thr_idx, shm_output_indexes, d_output_number);

			if (output_idx >= 0)
			{
				d_output_data[output_idx] = new_polarization;
			}
		} 
		else
		{
			d_next_polarization[thr_idx] = d_polarization[thr_idx];
		}

	}
}
   
extern "C"
void launch_bistable_simulation(
	double *h_polarization,
	double *h_Ek,
	int *h_cell_clock,
	int *h_neighbours,
	int cells_number,
	int neighbours_number,
	int number_of_samples,
	int max_iterations,
	int *input_indexes,
	int input_number,
	int *output_indexes,
	int output_number,
	double clock_prefactor_d,
	double clock_shift_d,
	double clock_low_d,
	double clock_high_d, 
	int input_values_number,
	char *input_values,
	double tolerance_d,
	double ***output_traces
	) //if input_values_number == -1 then EXHAUSTIVE
{


	// Variables
	double *d_next_polarization, *d_polarization, *d_Ek;
	int *d_neighbours, *d_cell_clock, *d_input_indexes, *d_output_indexes;
	int i,j,stable;
	int *d_stability, *h_stability;
	int count;
	int k;
	double *d_output_data;
	double *h_output_data;
	double clock_prefactor = (double)clock_prefactor_d;
	double clock_shift = (double)clock_shift_d;
	double clock_low = (double)clock_low_d;
	double clock_high = (double)clock_high_d;
	double tolerance = (double)tolerance_d;
	//int random_cell;

	/**printf("\nentrato nella launch gay!\n");
	printf("\ntesting launch parameters:\n cells_number = %d\n neighbours_number = %d \n number_of_samples = %d\n max_iterations = %d\n, tolerance = %f\ninput_values_number = %d\npref: %g, shift: %f, low: %f, high: %f\n",cells_number, neighbours_number, number_of_samples, max_iterations, (double)tolerance, input_values_number,clock_prefactor,clock_prefactor_d,clock_shift,clock_low,clock_high);**/
	//printf("output_number = %d, output_indexes[0]= %d", output_number , output_indexes[0]);


	h_output_data = (double *) malloc(sizeof(double) * output_number);
	h_stability = (int *)malloc(sizeof(int)*cells_number);
	for (i=0;i<cells_number;i++) h_stability[i] = 1;

	// Set GPU Parameters


	dim3 threads (BLOCK_DIM);
	dim3 grid (ceil ((double)cells_number/BLOCK_DIM));

	// Set Devices
	cudaSetDevice (cutGetMaxGflopsDeviceId());
	//cudaPrintfInit ();

	//starting timer
	timespec startTime, endTime;
	clock_gettime(CLOCK_REALTIME, &startTime);

	
	// Initialize Memory
	cutilSafeCall (cudaMalloc ((void**)&d_output_data, output_number * sizeof(double)));
	cutilSafeCall (cudaMalloc ((void**)&d_next_polarization, cells_number * sizeof(double)));
	cutilSafeCall (cudaMalloc ((void**)&d_polarization, cells_number * sizeof(double))); 
	cutilSafeCall (cudaMalloc ((void**)&d_Ek, sizeof(double)*neighbours_number*cells_number));
	cutilSafeCall (cudaMalloc ((void**)&d_cell_clock, cells_number * sizeof(int)));
	cutilSafeCall (cudaMalloc ((void**)&d_neighbours, sizeof(int)*neighbours_number*cells_number));
	cutilSafeCall (cudaMalloc ((void**)&d_input_indexes, sizeof(int)*input_number));
	cutilSafeCall (cudaMalloc ((void**)&d_output_indexes, sizeof(int)*output_number));
	cutilSafeCall (cudaMalloc ((void**)&d_stability, sizeof(int)*cells_number));

	// Set Memory

	cutilSafeCall (cudaMemcpy (d_next_polarization, h_polarization, cells_number * sizeof(double), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_polarization, h_polarization, cells_number * sizeof(double), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_Ek, (double *)h_Ek, sizeof(double) * neighbours_number * cells_number, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_cell_clock, h_cell_clock, cells_number * sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_neighbours, h_neighbours, sizeof(int) * neighbours_number * cells_number, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_input_indexes, input_indexes, sizeof(int)*input_number, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_output_indexes, output_indexes, sizeof(int)*output_number, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpy (d_stability, h_stability, sizeof(int)*cells_number, cudaMemcpyHostToDevice));

	cutilSafeCall (cudaMemcpyToSymbol("d_clock_prefactor", &(clock_prefactor), sizeof(double), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_clock_shift", &(clock_shift), sizeof(double), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_cells_number", &(cells_number), sizeof(int), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_neighbours_number", &(neighbours_number), sizeof(int), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_input_number", &(input_number), sizeof(int), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_output_number", &(output_number), sizeof(int), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_number_of_samples", &(number_of_samples), sizeof(double), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_clock_low", &(clock_low), sizeof(double), 0, cudaMemcpyHostToDevice));
	cutilSafeCall (cudaMemcpyToSymbol("d_clock_high", &(clock_high), sizeof(double), 0, cudaMemcpyHostToDevice));
	
	//srand(time(0));


	for (j = 0; j < number_of_samples ; j++)
	{

		stable = 0;

		update_inputs<<< grid, threads >>> (d_polarization, d_input_indexes, j);
		cudaThreadSynchronize ();

		//random_cell = (int)((double)rand()/(double)RAND_MAX*cells_number);
		
		// In each sample...
		for (i = 0; i < max_iterations && !stable; i++)
		{
			// Launch Kernel
			bistable_kernel<<< grid, threads >>> (d_polarization, d_next_polarization, d_cell_clock, d_Ek, d_neighbours, j, d_output_indexes, d_stability, tolerance, d_output_data);

		//	for (count = 0; count<cells_number; count++) printf("%d",h_stability[count]);
		//	printf("\n");

			// Wait Device
			cudaThreadSynchronize ();
			  
			cutilSafeCall (cudaMemcpy (h_stability, d_stability, cells_number*sizeof(int), cudaMemcpyDeviceToHost));

			count = 0;
			while (count<cells_number && h_stability[count] != 0) count++;
			if (count < cells_number) stable = 0;
			else stable = 1;
		  
	
			// Set Memory for the next iteration
			cutilSafeCall (cudaMemcpy (d_polarization, d_next_polarization, cells_number * sizeof(double), cudaMemcpyDeviceToDevice));
		}

		// Get desidered iteration results from GPU
		cutilSafeCall (cudaMemcpy (h_output_data, d_output_data, output_number * sizeof(double), cudaMemcpyDeviceToHost));

		for (k=0;k<output_number;k++)
		{
			 printf("%e\n", h_output_data[k]); //maybe %lf now that we use double?
			(*output_traces)[k][j] = h_output_data[k];
		}
	
		if(j%100 == 0) fprintf(stderr,"Simulating: %d % \titerations: %d\n", (j*100/number_of_samples), i);

		

	}
	//cudaPrintfDisplay(stdout, true);
	//cudaPrintfEnd();




	// Free-up resources
	cudaFree(d_output_data);
	cudaFree(d_next_polarization);
	cudaFree(d_input_indexes);
	cudaFree(d_output_indexes);
	cudaFree(d_polarization);
	cudaFree(d_cell_clock);
	cudaFree(d_stability);
	cudaFree(d_Ek);
	cudaFree(d_neighbours);


	//get time result	
	clock_gettime(CLOCK_REALTIME, &endTime);
		timespec temp;
	if ((endTime.tv_nsec-startTime.tv_nsec)<0)
	{
		temp.tv_sec = endTime.tv_sec-startTime.tv_sec-1;
		temp.tv_nsec = 1000000000+endTime.tv_nsec-startTime.tv_nsec;
	} 
	else
	{
		temp.tv_sec = endTime.tv_sec-startTime.tv_sec;
		temp.tv_nsec = endTime.tv_nsec-startTime.tv_nsec;
	}

	fprintf(stdout, "\tProcessing time1: %f (s)\n", (double)temp.tv_sec);

	fprintf(stdout, "\tProcessing time2: %f (ns)\n", (double)temp.tv_nsec);
}
