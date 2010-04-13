#include "../objects/QCADCell.h"

typedef struct {
	QCADCellFunction function;
	int clock;
	int number_of_neighbours;
  	int *neighbours; //index over the array of QCADCudaCells
  	double *Ek;
  	double polarization;
} QCADCudaCell;


