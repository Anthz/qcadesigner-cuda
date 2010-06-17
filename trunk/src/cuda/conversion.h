#ifndef _CONVERSION_H_
#define _CONVERSION_H_

#include "../objects/QCADCell.h"
#include "../bistable_simulation.h"

// void sorted_cells_to_CUDA_Structures_matrix(QCADCell ***sorted_cells, float **h_polarization, float **h_clock, float ***h_Ek,int ***h_neighbours, int number_of_cell_layers, int *number_of_cells_in_layer, int *neighbours_number);


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
	int* output_number,
	DESIGN *design
	);


#endif /* _CONVERSION_H_ */
