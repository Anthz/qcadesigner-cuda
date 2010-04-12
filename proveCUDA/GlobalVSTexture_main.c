#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_CELLS 10000
#define NUM_NEIGH_FIX 10
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


//-- Generates a psuedo-random integer between min and max
int randint(int min, int max)
{
   return (rand()%(int)(((max)+1)-(min)))+(min);
}


void computeGold (float *result, float *polarization, int *neighbours, int result_dim, int polarization_dim, int neighbours_dim) {
  
   int i, j, num_neighbours;
   float my_polarization, neighbours_polarization;

   num_neighbours = neighbours_dim/polarization_dim;

   for (i = 0; i < polarization_dim; i++)
   {
      my_polarization = polarization[i];
      neighbours_polarization = 0;

      for (j = 0; j < num_neighbours; j++)
	 neighbours_polarization += polarization[ neighbours[i*num_neighbours + j] ];

      result[i] = my_polarization * neighbours_polarization;
   }
}


int main () {
   
   int i, j;
   float polarization[NUM_CELLS], global_result[NUM_CELLS], texture_result[NUM_CELLS];
   int neighbours[NUM_CELLS*NUM_NEIGH_FIX];

   // Generate random polarizations
   for (i = 0; i < NUM_CELLS; i++)
      polarization[i] = randfloat(0, 10);

   // Generate random neighbourhood
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 neighbours[i*NUM_NEIGH_FIX + j] = randint(0, NUM_CELLS-1);

#ifdef __DEBUG_MODE__
   FILE *fp;

   fp = fopen("log_polarizations.txt", "w");
   for (i = 0; i < NUM_CELLS; i++)
   {
      fprintf(fp, "%5d\t||\t%03.04f\t||\t", i, polarization[i]);
      for (j = 0; j < NUM_NEIGH_FIX; j++)
      {
	 fprintf(fp, "%04d\t", neighbours[i*NUM_NEIGH_FIX + j]);
      }
      fprintf(fp, "\n");
   }
   fclose(fp);

   // Check Consistency
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 if ( !(0 <= neighbours[i*NUM_NEIGH_FIX + j] && neighbours[i*NUM_NEIGH_FIX + j] < NUM_CELLS) )
	 {
	    printf ("Consistency check on neighbours failed.\n");
	    return -1;
	 }
   printf ("Consistency check on neighbours OK.\n");
#endif

   launchGlobalKernel (global_result, polarization, neighbours, NUM_CELLS, NUM_CELLS, NUM_CELLS*NUM_NEIGH_FIX);

   launchTextureKernel (texture_result, polarization, neighbours, NUM_CELLS, NUM_CELLS, NUM_CELLS*NUM_NEIGH_FIX);

#ifdef __DEBUG_MODE__
   float gold_result[NUM_CELLS];

   computeGold(gold_result, polarization, neighbours, NUM_CELLS, NUM_CELLS, NUM_CELLS*NUM_NEIGH_FIX);

   fp = fopen("log_global_results.txt", "w");
   for (i = 0; i < NUM_CELLS; i++)
   {
      fprintf(fp, "%f\t||\t%f\n", gold_result[i], global_result[i]);
      if (fabs(gold_result[i] - global_result[i]) > 0.1f)
      {
	 printf("Verification Failed (Global Memory):\n\tgold_result[%d] = %f\n\tglobal_result[%d] = %f\n\n", i, gold_result[i], i, global_result[i]);
	 close(fp);
         return -1;
      } 
    }
    fclose(fp);
    printf("Verification Success (Global Memory)!\n");

   fp = fopen("log_texture_results.txt", "w");
   for (i = 0; i < NUM_CELLS; i++)
   {
      fprintf(fp, "%f\t||\t%f\n", gold_result[i], texture_result[i]);
      if (fabs(gold_result[i] - texture_result[i]) > 0.1f)
      {
	 printf("Verification Failed (Texture Memory):\n\tgold_result[%d] = %f\n\tglobal_result[%d] = %f\n\n", i, gold_result[i], i, texture_result[i]);
	 close(fp);
         return -1;
      } 
    }
    fclose(fp);
    printf("Verification Success (Texture Memory)!\n");

#endif

   return 0;

}



