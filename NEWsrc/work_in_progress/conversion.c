
int num_of_neighbours; //to be initialized
int tmp=0;

for(iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
	tmp += number_of_cells_in_layer[iLayer];
}

double* polarization = (double*) malloc(tmp*(sizeof(double)));
double** kink_enegry = (double**) malloc(tmp*num_of_neighbours*(sizeof(double*)));
int** neighbouring_index = (int**) malloc(tmp*num_of_neighbours*(sizeof(int*)));

//fare le malloc degli array..
/*Logica riempimento matrici:
Matrice kink enegy
ID|#delvicino 0,1,2 -->
1?|Ek|Ek|Ek|
2?|Ek|Ek|Ek|

matrice degli id.
ID|#delvicino 0,1,2 -->
1?|ID|ID|ID|
2?|ID|ID|ID|

array polarizazione
ID|polariz
1?|pol
2?|pol

TODO: controllare l'utilizzo del campo ID della cella. 
controllare allocazione
uniformare la struttura di array di polarizzioni alle altre di matrici? (introducendo ridondanza?)???
testare testare testare. 
*/



for(iLayer = 0; iLayer < number_of_cell_layers; iLayer++){
	for (iCell = 0; iCell < number_of_cells_in_layer[iLayer];iCell++){
		polarization[sorted_cells[iLayer][iCell]->id] = sorted_cells[iLayer][iCell]->cell_model->polarization;
		for (iNeighbour = 0; iNeighbour < sorted_cells[iLayer][iCell]->cell_model->number_of_neighbours; iNeighbour++)
			kink_energy[sorted_cells[iLayer][iCell]->id][iNeighbour]=sorted_cells[iLayer][iCell]->cell_model->Ek[iNeighbour];
			neighbouring_index[sorted_cells[iLayer][iCell]->id][iNeighbour]= neighbouring_index[sorted_cells[iLayer][iCell]->neighbours[q]->id;
	}
}
