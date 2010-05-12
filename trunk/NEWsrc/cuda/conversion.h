#ifndef _CONVERSION_H_
#define _CONVERSION_H_

#include "objects/QCADCell.h"
#include "bistable_simulation.h"

extern void sorted_cells_to_CUDA_Structures(QCADCell ***sorted_cells, float **h_polarization, float **h_clock, float ***h_Ek,int ***h_neighbours, int cells_number, int number_of_cell_layers, int *number_of_cells_in_layer);

#endif /* _CONVERSION_H_ */