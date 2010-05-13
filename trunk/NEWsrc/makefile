CC = nvcc
GLIB=`pkg-config gtk+-2.0 --cflags --libs glib-2.0`
FLAGS=-g

SOURCES = main_batch_sim.c design.c fileio.c fileio_helpers.c exp_array.c objects/QCADDesignObject.c objects/QCADLayer.c objects/QCADCell.c objects/QCADCompoundDO.c objects/QCADDOContainer.c objects/object_helpers.c objects/QCADSubstrate.c objects/QCADStretchyObject.c objects/QCADLabel.c generic_utils.c vector_table.c bistable_simulation.c simulation_data.c simulation.c coherence_vector.c cuda/conversion.c

all: QCADesigner

clean: 
	rm -f *.o
	rm QCADesigner

QCADesigner: ${SOURCES}
	${CC} ${GLIB} ${FLAGS} -o QCADesigner ${SOURCES}

