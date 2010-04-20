#include "objects/QCADCell.h"
#include "bistable_simulation.h"
#include <stdlib.h>

 QCADCell *tmp;

extern void sorted_cells_to_CUDA_Structures(QCADCell ***sorted_cells, float **h_polarization, float **h_clock, float ***h_Ek, int ***h_neighbours, int cells_number, int number_of_cell_layers, int *number_of_cells_in_layer){
  
 // Allocate memory for all needed structures
 int i,iLayer, iCell, iNeighbour;
 int neighbours_number = 0;
 int counter = 0;

tmp = (QCADCell* )malloc(cells_number*sizeof(QCADCell));
 
 //init neighbours_number
 for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      if(neighbours_number < ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours){
//	printf("cella %d,%d, vicini %d\n",iLayer,iCell,((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours);
	neighbours_number = ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours;
      } 
    }
 }



 *h_polarization = (float*) malloc(cells_number*sizeof(float));
 *h_clock = (float*) malloc(cells_number*sizeof(float));
 *h_Ek = (float**) malloc(cells_number*sizeof(float*));
 for( i = 0; i < cells_number;i++)
   (*h_Ek)[i]= (float*) malloc(neighbours_number*sizeof(float));
 *h_neighbours = (int**) malloc(cells_number*sizeof(int*));
 for( i = 0; i < cells_number;i++)
   (*h_neighbours)[i]= (int*) malloc(neighbours_number*sizeof(int)); 



//fill tmp
  for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
       tmp[counter] = (QCADCell)*sorted_cells[iLayer][iCell];
    }
 }


 //fill structures 
  for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      (*h_polarization)[counter] = ((bistable_model*)(sorted_cells[iLayer][iCell]->cell_model))->polarization;
      (*h_clock)[counter] = sorted_cells[iLayer][iCell]->cell_options.clock;
//    (*h_polarization)[sorted_cells[iLayer][iCell]->id] = ((bistable_model*)(sorted_cells[iLayer][iCell]->cell_model))->polarization;
//    (*h_clock)[sorted_cells[iLayer][iCell]->id] = sorted_cells[iLayer][iCell]->cell_options.clock;
//	printf("hi\n");
      for ( iNeighbour = 0; iNeighbour < ((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->number_of_neighbours;iNeighbour++){
	(*h_Ek)[counter][iNeighbour]=((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->Ek[iNeighbour];
	//TODO: find the right neighbour.
	(*h_neighbours)[counter][iNeighbour]= position_in_CUDA_array(((QCADCell*)((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->neighbours[iNeighbour])->id,cells_number);
//	(*h_Ek)[sorted_cells[iLayer][iCell]->id][iNeighbour]=((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->Ek[iNeighbour];
//	(*h_neighbours)[sorted_cells[iLayer][iCell]->id][iNeighbour]= ((QCADCell*)((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->neighbours[iNeighbour])->id;
	}
       counter++;
      }
    }
printf("final counter:%d",counter);
  }

int position_in_CUDA_array(int cell_id, int lenght){
   int i;
   for(i=0;i<lenght;i++){
      if(tmp[i].id = cell_id) return i;
   }
}

