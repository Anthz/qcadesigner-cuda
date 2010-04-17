#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "structs.h"

#define NUM_CELLS 100
#define NUM_NEIGH_FIX 4
#define __DEBUG_MODE__


//-- Generates a psuedo-random float between min and max
float randfloat (float min, float max)
{
   if (min>max) 
   {
      return rand()/((float)RAND_MAX+1)*(min-max)+max;    
   } 
   else 
   {
      return rand()/((float)RAND_MAX+1)*(max-min)+min;
   }
}


//------------------------------------------------------------------------------------ 


//-- Generates a psuedo-random integer between min and max
int randint(int min, int max)
{
   return (rand()%(int)(((max)+1)-(min)))+(min);
}


//------------------------------------------------------------------------------------


int main () 
{
   
   int i, j;
   float polarization[NUM_CELLS], clock[NUM_CELLS], lambda_x[NUM_CELLS], lambda_y[NUM_CELLS], lambda_z[NUM_CELLS];
   float Ek[NUM_CELLS][NUM_NEIGH_FIX];
   int neighbours[NUM_CELLS][NUM_NEIGH_FIX];
   int iterations = 1;
   CUDA_coherence_OP options;
   CUDA_coherence_optimizations optimization_options;

   // Generate random polarizations, clock, lambda
   for (i = 0; i < NUM_CELLS; i++)
   {
      polarization[i] = randfloat(0, 10);
      clock[i] = randfloat(0, 1);
      lambda_x[i] = randfloat(0, 1);
      lambda_y[i] = randfloat(0, 1);
      lambda_z[i] = randfloat(0, 1);
   }

   // Generate random Ek
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 Ek[i][j] = randfloat(0, 10);
      
   // Generate random neighbours
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 neighbours[i][j] = randint(0, NUM_CELLS-1);

   // Generate Random Options
   options.T = randfloat(0, 10);
   options.relaxation = randfloat(0, 10);
   options.time_step = randfloat(0, 10);
   options.duration = randfloat(0, 10);
   options.clock_high = randfloat(0, 10);
   options.clock_low = randfloat(0, 10);
   options.clock_shift = randfloat(0, 10);
   options.clock_amplitude_factor = randfloat(0, 10);
   options.radius_of_effect = randfloat(0, 10);
   options.epsilonR = randfloat(0, 10);
   options.layer_separation = randfloat(0, 10);
   options.algorithm = 1;

   // Generate Random Optimization Options
   optimization_options.clock_prefactor = randfloat(0, 10);
   optimization_options. clock_shift = randfloat(0, 10);
   optimization_options.four_pi_over_number_samples = randfloat(0, 10);
   optimization_options.two_pi_over_number_samples = randfloat(0, 10);
   optimization_options.hbar_over_kBT = randfloat(0, 10);

#ifdef __DEBUG_MODE__
   FILE *fp;

   fp = fopen("log/log.txt", "w");
   fprintf(fp,"Index\t||\tPolarization\t||\tClock\t||\tLambda\t||\tNeighbours\t||\tEk\n");
   for (i = 0; i < NUM_CELLS; i++)
   {
      fprintf(fp, "%5d\t||\t%03.04f\t||\t%f\t||\t%f %f %f\t||\t", i, polarization[i], clock[i], lambda_x[i], lambda_y[i], lambda_z[i]);
      for (j = 0; j < NUM_NEIGH_FIX; j++)
      {
	 fprintf(fp, "%04d ", neighbours[i][j]);
      }
      
      fprintf(fp, "\t||\t");

      for (j = 0; j < NUM_NEIGH_FIX; j++)
      {
	 fprintf(fp, "%f ", Ek[i][j]);
      }

      fprintf(fp, "\n");
   }
   fclose(fp);

   // Check Consistency
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 if ( !(0 <= neighbours[i][j] && neighbours[i][j] < NUM_CELLS) )
	 {
	    printf ("Consistency check on neighbours failed.\n");
	    return -1;
	 }
   printf ("Consistency check on neighbours OK.\n");
#endif

   // Launch Simulation
   launch_coherence_vector_simulation (polarization, clock, lambda_x, lambda_y, lambda_z, Ek, neighbours, NUM_CELLS, NUM_NEIGH_FIX, iterations, &options, &optimization_options);

   for (i = 0; i < NUM_CELLS; i++)
      printf("polarization[%d] = %f\n", i, polarization);

   
   return 0;

}



