#include<stdio.h>
#include<stdlib.h>
#include "coloring.h"

#define MAX(a,b) a > b ? a : b


int *node_colors;
int *nb_colors;

int color_graph(int *array_graph, int node_num, int max_num_arcs, int **cell_colors, int *num_colors)
{
	
	int i, j, k, nb_num;
	int nb_idx;
	int color_chosen;
	int found;
	(*num_colors)=1;
		
	node_colors = (*cell_colors) = (int*)calloc(node_num,sizeof(int));	
	nb_colors = (int*)malloc(sizeof(int)*max_num_arcs);	
	
	node_colors[0] = 1;
	
	for (i = 1; i < node_num; i++)
	{
		color_chosen = 1;
		nb_num = 0;
		found=0;
		nb_idx = array_graph[i];
		if (nb_idx == -1) node_colors[i] = color_chosen; //no neighbours
		else
		{
			for (j=0; j<max_num_arcs && nb_idx!=-1; j++)
			{
				nb_idx = array_graph[i+node_num*j];
				if (nb_idx!=-1 && array_graph[nb_idx]!=-1) //if I have a neighbor and this neighbor is not a fixed or input
				{
					nb_num++;
					nb_colors[j] = node_colors[nb_idx];
				}
			}
			while (found==0)
			{
				found = 1;
				for (j=0; j<nb_num; j++)
				{
					//printf("%d",nb_colors[j]);
					if (color_chosen == nb_colors[j]) found = 0;
				}
				//printf("\n");
				color_chosen++;
			}
			node_colors[i] = color_chosen-1;
		}
		(*num_colors)= MAX(color_chosen-1,(*num_colors));
	}
	return 0;
}


