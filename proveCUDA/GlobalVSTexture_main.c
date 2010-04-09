#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_CELLS 10000
#define NUM_NEIGH_FIX 10
//#define __DEBUG_MODE__


//-- Generates a psuedo-random double between min and max
double randdouble (double min, double max)
{
   if (min>max) 
   {
      return rand()/((double)RAND_MAX+1)*(min-max)+max;    
   } 
   else 
   {
      return rand()/((double)RAND_MAX+1)*(max-min)+min;
   }
} 


//-- Generates a psuedo-random integer between min and max
int randint(int min, int max)
{
   return (rand()%(int)(((max)+1)-(min)))+(min);
}


int main () {
   
   int i, j;
   double polarization[NUM_CELLS];
   int neighbours[NUM_CELLS*NUM_NEIGH_FIX];

   // Generate random polarizations
   for (i = 0; i < NUM_CELLS; i++)
      polarization[i] = randdouble(0, 100);

   // Generate random neighbourhood
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 neighbours[i*NUM_NEIGH_FIX + j] = randint(0, NUM_CELLS);

#ifdef __DEBUG_MODE__
   for (i = 0; i < NUM_CELLS; i++)
   {
      printf("%d  ||  %.6g  ||  ", i, polarization[i]);
      for (j = 0; j < NUM_NEIGH_FIX; j++)
      {
	 printf("%.3d  ", neighbours[i*NUM_NEIGH_FIX + j]);
      }
      printf("\n");
   }
   printf("\n");

   // Check Consistency
   for (i = 0; i < NUM_CELLS; i++)
      for (j = 0; j < NUM_NEIGH_FIX; j++)
	 if ( !(0 <= neighbours[i*NUM_NEIGH_FIX + j] < NUM_CELLS) )
	 {
	    printf ("Consistency check on neighbours failed.\n");
	    return -1;
	 }
   printf ("Consistency check on neighbours OK.\n");
#endif

   launchGlobalKernel (polarization, neighbours, NUM_CELLS, NUM_CELLS*NUM_NEIGH_FIX);

   launchTextureKernel (polarization, neighbours, NUM_CELLS, NUM_CELLS*NUM_NEIGH_FIX);

   return 0;

}



