#ifndef _QCADcuda_H_
#define _QCADcuda_H_
#include "../objects/QCADCell.h"

typedef struct {
	QCADCellFunction function;
	int clock;
	int number_of_neighbours;
  	int *neighbours; //index over the array of QCADCudaCells
  	double *Ek;
  	double polarization;
} QCADCudaCell;


#endif /* _QCADcuda_H_ */