#include <stdio.h>
 
int main(){
   
   int i;
   double cpu_size[5];

   double gpu_size[5];

   cpu_size[0] = sizeof(short);
   cpu_size[1] = sizeof(int);
   cpu_size[2] = sizeof(float);
   cpu_size[3] = sizeof(double);  
   cpu_size[4] = sizeof(long int); 

   launchKernel (gpu_size);

   printf("\n         |   CPU  |   GPU  \n");
   printf("---------------------------\n");
   printf("Short    |   %g   |   %g   \n", cpu_size[0]*8, gpu_size[0]*8);
   printf("Int      |   %g   |   %g   \n", cpu_size[1]*8, gpu_size[1]*8);
   printf("Float    |   %g   |   %g   \n", cpu_size[2]*8, gpu_size[2]*8);
   printf("Double   |   %g   |   %g   \n", cpu_size[3]*8, gpu_size[3]*8);  
   printf("Long Int     |   %g   |   %g   \n\n", cpu_size[4]*8, gpu_size[4]*8); 

   return 0;

}
