#include "../objects/QCADCell.h"
#include "../bistable_simulation.h"
#include <stdlib.h>

 int *tmp;

int position_in_CUDA_array(int cell_id, int lenght){
   int i;

   for(i=0;i<lenght;i++){
      if(tmp[i] == cell_id){
	 return i;
	}
   }
}


void sorted_cells_to_CUDA_Structures_array(
	QCADCell ***sorted_cells,
	double **h_polarization, 
	int **h_cell_clock,
	double **h_Ek, 
	int **h_neighbours,
	int number_of_cell_layers,
	int *number_of_cells_in_layer,
	int* neighbours_number,
	int** input_indexes,
	int* input_number,
	int** output_indexes,
	int* output_number
	)
{
  
 // Allocate memory for all needed structures
 int i,iLayer, iCell, iNeighbour;
 *neighbours_number = 0;
 int counter = 0;
 int input_counter = 0;
 int output_counter = 0;
 int cells_number = 0;
 int j, ambros;

/*h_clock_data = (double *) malloc(number_of_samples * 4 * sizeof(double));

for(j = 0; j < number_of_samples; j++)
  {
    for(ambros=0; ambros<4;ambros++){
      (*h_clock_data)[j * 4 + ambros] = (double)clock_source[ambros].data[j];
    }
  } */



for(i = 0; i < number_of_cell_layers; i++)
    cells_number+= number_of_cells_in_layer[i];


tmp = (int*)malloc(cells_number*sizeof(int));
 
 //init neighbours_number
 for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      if(*neighbours_number < ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours){
	*neighbours_number = ((bistable_model *)sorted_cells[iLayer][iCell]->cell_model)->number_of_neighbours;
      } 
    }
 }


 *h_polarization = (double*) malloc(cells_number*sizeof(double));
 *h_cell_clock = (int*) malloc(cells_number*sizeof(double));
 *h_Ek = (double*) malloc(*neighbours_number*cells_number*sizeof(double));
 *h_neighbours = (int*) malloc(*neighbours_number*cells_number*sizeof(int));


//fill tmp and count number of inputs
  for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
       tmp[counter] = sorted_cells[iLayer][iCell]->id;
	if (sorted_cells[iLayer][iCell]->cell_function == QCAD_CELL_INPUT){
	   input_counter++;
	}
	else if (sorted_cells[iLayer][iCell]->cell_function == QCAD_CELL_OUTPUT){
	   output_counter++;
	}
	counter++;
    }
 }


*input_number = input_counter;
*output_number = output_counter;
 
*input_indexes = (int *)malloc(*input_number * sizeof(int));
*output_indexes = (int *)malloc(*output_number * sizeof(int));


counter =0;
input_counter = 0;
output_counter = 0;
//init neighbours values to -1 and fill input_indexes
  for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      if(sorted_cells[iLayer][iCell]->cell_function == QCAD_CELL_INPUT){
	(*input_indexes)[input_counter] = counter;
	input_counter++;
	}
      else if(sorted_cells[iLayer][iCell]->cell_function == QCAD_CELL_OUTPUT){
	(*output_indexes)[output_counter] = counter;
	output_counter++;
	}
      for(i=0;i<*neighbours_number;i++){
       (*h_neighbours)[counter*(*neighbours_number) + i] = -1;
	(*h_Ek)[counter*(*neighbours_number) + i] = 0.0;
      }
    counter++;
    }
 }


counter =0;

 //fill structures 
  for( iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
    for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
      (*h_polarization)[counter] = (double)((bistable_model*)(sorted_cells[iLayer][iCell]->cell_model))->polarization;
      (*h_cell_clock)[counter] = sorted_cells[iLayer][iCell]->cell_options.clock;
      for ( iNeighbour = 0; iNeighbour < ((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->number_of_neighbours;iNeighbour++){
	(*h_Ek)[counter + iNeighbour*cells_number]=(double)((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->Ek[iNeighbour];
	if(sorted_cells[iLayer][iCell]->cell_function != QCAD_CELL_INPUT && sorted_cells[iLayer][iCell]->cell_function != QCAD_CELL_FIXED)
	  (*h_neighbours)[counter + iNeighbour*cells_number]= position_in_CUDA_array(((QCADCell*)((bistable_model *)(sorted_cells[iLayer][iCell]->cell_model))->neighbours[iNeighbour])->id,cells_number);

	}
       counter++;
      }
    }
 free(tmp);


}
