//////////////////////////////////////////////////////////
// QCADesigner                                          //
// Copyright 2002 Konrad Walus                          //
// All Rights Reserved                                  //
// Author: Konrad Walus                                 //
// Email: qcadesigner@gmail.com                         //
// WEB: http://qcadesigner.ca/                          //
//////////////////////////////////////////////////////////
//******************************************************//
//*********** PLEASE DO NOT REFORMAT THIS CODE *********//
//******************************************************//
// If your editor wraps long lines disable it or don't  //
// save the core files that way. Any independent files  //
// you generate format as you wish.                     //
//////////////////////////////////////////////////////////
// Please use complete names in variables and fucntions //
// This will reduce ramp up time for new people trying  //
// to contribute to the project.                        //
//////////////////////////////////////////////////////////
// This file was contributed by Gabriel Schulhof        //
// (schulhof@atips.ca).                                 //
//////////////////////////////////////////////////////////
// Contents:                                            //
//                                                      //
// A batch mode simulator aimed specifically at         //
// verifying outputs against their corresponding        //
// inputs.                                              //
//                                                      //
//////////////////////////////////////////////////////////



#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <glib.h>
#include <time.h>
#include "fileio.h"
#include "design.h"
//#include "graph_dialog.h"
#include "global_consts.h"
#include "simulation.h"
#include "simulation_data.h"
#include "coherence_vector.h"
//#include "graph_dialog_widget_data.h"
#include "bistable_simulation.h"

extern bistable_OP bistable_options ;
extern bistable_OP bistable_options ;
extern coherence_OP coherence_options ;
//
static void randomize_design_cells (GRand *rnd, DESIGN *design, double dMinRadius, double dMaxRadius) ;
//
//static int determine_success (HONEYCOMB_DATA *hcIn, HONEYCOMB_DATA *hcOut) ;
//
static void parse_cmdline (int argc, char **argv, int *sim_engine, char **pszSimOptsFName, char **pszFName, int *number_of_sims, double *dTolerance, char **pszFileSave) ;

int main (int argc, char **argv)
  {
  static char *pszSimEngine = NULL, *pszSimOptsFName = NULL, *pszFName = NULL ;
  int number_of_sims = -1 ;

  int sim_engine = BISTABLE ;
  int Nix, Nix1, Nix2 ;
  
  DESIGN *design = NULL, *working_design = NULL ;
  simulation_data *sim_data = NULL ;
  GRand *rnd = NULL ;
//  GdkColor clr = {0, 0, 0, 0} ;
/*  EXP_ARRAY *input_hcs = NULL, *output_hcs = NULL, *hc_ar = NULL ;*/
/*  HONEYCOMB_DATA *hc = NULL ;*/
/*  BUS *bus = NULL ;*/
//  struct TRACEDATA *the_trace = NULL ;
  double dTolerance = -1.0 ;
  EXP_ARRAY *icSuccesses = NULL ;
  int icOutputBuses = 0 ;
  char *pszFileSaveOut=(char*)malloc(1000*sizeof(char));
  char *pszFileSaveBin=(char*)malloc(1000*sizeof(char));
  long int lTime = 0;
  char *sTime=(char*)malloc(100*sizeof(char));
  struct timespec temp, startTime, endTime;
  double sim_time;

  //intial time for simulation.
  clock_gettime(CLOCK_REALTIME, &startTime);
  fprintf(stderr,"Initialization...");
  fflush(stderr);

  parse_cmdline (argc, argv, &sim_engine, &pszSimOptsFName, &pszFName, &number_of_sims, &dTolerance, &pszFileSaveOut) ;

//#ifdef GTK_GUI
//  gtk_init (&argc, &argv) ;
//#else
  g_type_init () ;
//#endif /* def GTK_GUI */

  if (pszSimEngine != NULL)
    sim_engine =
      !strncmp (pszSimEngine, "BISTABLE", sizeof ("BISTABLE") - 1)
        ? BISTABLE
        : !strncmp (pszSimEngine, "COHERENCE_VECTOR", sizeof ("COHERENCE_VECTOR") - 1)
          ? COHERENCE_VECTOR
          : BISTABLE /* default */ ;

  if (!open_project_file (pszFName, &design))
    {
    fprintf (stderr, "Failed to open the circuit file %s!\n", pszFName) ;
    return 3 ;
    }
  else
  if (NULL == design)
    {
    fprintf (stderr, "Failed to open the circuit file %s!\n", pszFName) ;
    return 3 ;
    }


  if (BISTABLE == sim_engine)
    {
    bistable_OP *bo = NULL ;

    if (NULL == (bo = open_bistable_options_file (pszSimOptsFName)))
      {
      fprintf (stderr, "Failed to open simulation options file %s!\n", pszSimOptsFName) ;
      return 2 ;
      }
/*    bistable_options_dump (bo, stderr) ;*/
    memcpy (&bistable_options, bo, sizeof (bistable_OP)) ;
    }
  else
  if (COHERENCE_VECTOR == sim_engine)
    {
    coherence_OP *co = NULL ;

    if (NULL == (co = open_coherence_options_file (pszSimOptsFName)))
      {
      fprintf (stderr, "Failed to open simulation options file %s!\n",pszSimOptsFName) ;
      return 2 ;
      }
/*    coherence_options_dump (co, stderr) ;*/
    memcpy (&coherence_options, co, sizeof (coherence_OP)) ;
    }

  fprintf (stderr, " done!\n") ;
  printf ("Running %d simulations with a radial tolerance of %lf\n", number_of_sims, dTolerance) ;


  rnd = g_rand_new () ;

  icSuccesses = exp_array_new (sizeof (int), 1) ;
  for (Nix = 0 ; Nix < design->bus_layout->buses->icUsed ; Nix++)
    if (QCAD_CELL_OUTPUT == exp_array_index_1d (design->bus_layout->buses, BUS, Nix).bus_function)
      icOutputBuses++ ;
  exp_array_insert_vals (icSuccesses, NULL, icOutputBuses, 1, 0) ;
  for (Nix = 0 ; Nix < icSuccesses->icUsed ; Nix++)
    exp_array_index_1d (icSuccesses, int, Nix) = 0 ;
	
	for (Nix = 0 ; Nix < number_of_sims ; Nix++)
	{
		
		fprintf (stdout, "Running simulation %d\n", Nix) ;
		if (NULL != (working_design = design_copy (design)))
		{

		   randomize_design_cells (rnd, working_design, 0.0, dTolerance) ;

		   if (NULL != (sim_data = run_simulation (sim_engine, EXHAUSTIVE_VERIFICATION, working_design, NULL)))
			{
				//final time for simulation.
				clock_gettime(CLOCK_REALTIME, &endTime);
				if ((endTime.tv_nsec-startTime.tv_nsec)<0)
				{
					temp.tv_sec = endTime.tv_sec-startTime.tv_sec-1;
					temp.tv_nsec = 1000000000+endTime.tv_nsec-startTime.tv_nsec;
				} 
				else
				{
					temp.tv_sec = endTime.tv_sec-startTime.tv_sec;
					temp.tv_nsec = endTime.tv_nsec-startTime.tv_nsec;
				}
				sim_time = (double)temp.tv_sec + (double)temp.tv_nsec/1000000000;
				printf("Total simulation time = %f seconds\n",sim_time);

				// saving data
				SIMULATION_OUTPUT sim_output;
				sim_output.sim_data = sim_data;
				sim_output.bus_layout = NULL;

				strcpy(pszFileSaveBin, pszFileSaveOut);

				FILE *file_out;
				#ifdef CUDA
				strcat(pszFileSaveOut, "-cuda.out\0");
				#else
				strcat(pszFileSaveOut, "-cpu.out\0");
				#endif
				//strcat(pszFileSaveOut, pszFName);
				file_out = fopen(pszFileSaveOut, "w");
				if (file_out == NULL)
				{
					printf("Output file doesn't exist.\n");
					return -1;
				}
				create_simulation_output_file_fp_cuda (file_out, &sim_output, bistable_options.delay); 
				fclose(file_out);
				
				FILE *file_bin;
				#ifdef CUDA
				strcat(pszFileSaveBin, "-cuda.bin\0");
				#else
				strcat(pszFileSaveBin, "-cpu.bin\0");
				#endif

				//strcat(pszFileSaveBin, pszFName);
				file_bin = fopen(pszFileSaveBin, "w");
				if (file_out == NULL)
				{
					printf("Output file doesn't exist.\n");
					return -1;
				}
				create_simulation_output_binary_cuda (file_bin, &sim_output,bistable_options.delay);
				fclose(file_bin);

				sim_data = simulation_data_destroy (sim_data) ;
			}
		working_design = design_destroy (working_design) ;
		}
	}

  for (Nix = 0 ; Nix < icSuccesses->icUsed ; Nix++)
    printf ("success_rate[%d] = %.2lf%%\n", Nix, (double)(exp_array_index_1d (icSuccesses, int, Nix)) / ((double)(number_of_sims)) * 100.0) ;

  g_rand_free (rnd) ;


  return 0 ;
  }

static void randomize_design_cells (GRand *rnd, DESIGN *design, double dMinRadius, double dMaxRadius)
  {
  double dRadius = -1.0, dAngle = 0.0 ;
  double dx = 0.0, dy = 0.0 ;
  GList *llItr = NULL, *llItrObj = NULL ;
  QCADLayer *layer = NULL ;

  if (NULL == rnd || NULL == design) return ;

  for (llItr = design->lstLayers ; llItr != NULL ; llItr = llItr->next)
    if (LAYER_TYPE_CELLS == (layer = QCAD_LAYER (llItr->data))->type)
      for (llItrObj = layer->lstObjs ; llItrObj != NULL ; llItrObj = llItrObj->next)
        if (NULL != llItrObj->data)
          {
          dRadius = g_rand_double_range (rnd, dMinRadius, dMaxRadius) ;
          dAngle = g_rand_double_range (rnd, 0, 2.0 * PI) ;

          dx = dRadius * cos (dAngle) ;
          dy = dRadius * sin (dAngle) ;

          qcad_design_object_move (QCAD_DESIGN_OBJECT (llItrObj->data), dx, dy) ;
          }
  }

static void parse_cmdline (int argc, char **argv, int *sim_engine, char **pszSimOptsFName, char **pszFName, int *number_of_sims, double *dTolerance, char** pszFileSave)
  {
  int icParms = 0 ;
  int Nix ;

  // defaults
  (*sim_engine) = BISTABLE ;

  for (Nix = 0 ; Nix < argc ; Nix++)
    {
    if (!strncmp (argv[Nix], "-f", 2))
      {
      if (++Nix < argc)
        {
        (*pszFName) = argv[Nix] ;
        icParms++ ;
        }
      }
    else
    if (!strncmp (argv[Nix], "-e", 2))
      {
      if (++Nix < argc)
        {
        (*sim_engine) =
          !strncmp (argv[Nix], "BISTABLE", 8)
            ? BISTABLE
            : !strncmp (argv[Nix], "COHERENCE_VECTOR", 16)
              ? COHERENCE_VECTOR
              : BISTABLE /* default */ ;
        icParms++ ;
        }
      }
    else
    if (!strncmp (argv[Nix], "-o", 2))
      {
      if (++Nix < argc)
        {
        (*pszSimOptsFName) = argv[Nix] ;
        icParms++ ;
        }
      }
    else
    if (!strncmp (argv[Nix], "-n", 2))
      {
      if (++Nix < argc)
        {
        (*number_of_sims) = atoi (argv[Nix]) ;
        icParms++ ;
        }
      }
    else
    if (!strncmp (argv[Nix], "-t", 2))
      {
      if (++Nix < argc)
        {
        (*dTolerance) = g_ascii_strtod (argv[Nix], NULL) ;
        icParms++ ;
        }
      }
    
    else
    if (!strncmp (argv[Nix], "-s", 4))
    {      
      if (++Nix < argc)
        {
        strcpy((*pszFileSave),argv[Nix]);
        icParms++ ;
        }
      }
    }

  if (icParms < 6)
    {
    printf (
      "Usage:\n"
      "batch_sim -f qca_file [-e [BISTABLE]|COHERENCE_VECTOR] -o engine_options_file -n number_of_simulations -t radial_tolerance -s outputfile\n") ;
    exit (1) ;
    }
  }

//static int determine_success (HONEYCOMB_DATA *hcdIn, HONEYCOMB_DATA *hcdOut)
//  {
//  int Nix ;
//  int idxIn = 0 ;
//  HONEYCOMB *hcIn = NULL, *hcOut = NULL ;
//
//  if (NULL == hcdIn || NULL == hcdOut) return 0 ;
//  if (NULL == hcdIn->arHCs || NULL == hcdOut->arHCs) return 0 ;
//  if (0 == hcdIn->arHCs->icUsed || 0 == hcdOut->arHCs->icUsed) return 0 ;
//
//  hcIn = &exp_array_index_1d (hcdIn->arHCs, HONEYCOMB, 0) ;
//
//  for (Nix = 0 ; Nix < hcdOut->arHCs->icUsed ; Nix++)
//    {
//    hcOut = &exp_array_index_1d (hcdOut->arHCs, HONEYCOMB, Nix) ;
//
//    // This output honeycomb may be contained withing the next input honeycomb
//    if (hcOut->idxBeg > hcIn->idxEnd)
//      {
//      if (++idxIn == hcdIn->arHCs->icUsed) return 0 ;
//      hcIn = &exp_array_index_1d (hcdIn->arHCs, HONEYCOMB, idxIn) ;
//      }
//
//    // The output honeycomb is not entirely contained within the input honeycomb
//    if (hcOut->idxBeg < hcIn->idxBeg || hcOut->idxEnd > hcIn->idxEnd)
//      return 0 ;
//
//    if (hcOut->value != hcIn->value)
//      return 0 ;
//    }
//
//  return 1 ;
//  }
