#include "bistable_simulation.h"
#include "conversion.h"
#include "objects/QCADCell.h"


//common variables
float *h_polarization, *h_clock, **h_Ek;
int **h_neighbours, cells_number=0,i,j;

extern void test_conversion(QCADCell*** sorted_cells, int number_of_cell_layers, int *number_of_cells_in_layer){
  
int iLayer, iCell, neighbours_number = 0;

  //find cells number
  for(i = 0; i < number_of_cell_layers; i++)
    cells_number+= number_of_cells_in_layer[i];

//    printf("total number of cells %d\n",cells_number);


  //allocate structures
  sorted_cells_to_CUDA_Structures(sorted_cells,&h_polarization,&h_clock,&h_Ek,&h_neighbours,cells_number, number_of_cell_layers, number_of_cells_in_layer);

 //init neighbours_number
 for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      if(neighbours_number < ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours)
	neighbours_number = ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours;
    }
 }

//  printf("\ncells: %d, layers: %d cells per layer 0: %d, neighbours: %d",cells_number,number_of_cell_layers, number_of_cells_in_layer[0],neighbours_number);
  //look at the structure.
 /* printf("Polarization:\n");
  for(i =0;i< cells_number;i++){
    printf("%d: %f\n",i,h_polarization[i]);
  }
  
  printf("\nClock:\n");
  for( i =0;i< cells_number;i++){
    printf("%d: %f\n",i,h_clock[i]);
  }
  /*
  printf("\nKink Energy:\n");
  for( i =0;i< cells_number;i++){
    printf("%d: ",i);
    for( j=0;j< neighbours_number;j++){
      printf("%f ",h_Ek[i][j]);
    }
    printf("\n");
  }*/
  
  printf("\nNeighbours:\n");
  for( i =0;i< cells_number;i++){
    printf("%d: ",i);
    for( j=0;j< neighbours_number;j++){
      printf("%d ",h_neighbours[i][j]);
    }
    printf("\n");
  }
//  system("PAUSE");
}

