CFLAGS = -Drestrict=__restrict__ -O3 -DGRAPH_GENERATOR_MPI -DREUSE_CSR_FOR_VALIDATION -I../aml -fcommon
LDFLAGS = -lpthread
MPICC = mpicc
PREPROCESSOR_FLAGS ?=
# -DDEBUGSTATS -DFORCE_INTERNODE_ONLY -DAGGR_intra=524288 -DAGGR=524288
PREPROCESSOR_FLAGS += -DCCUTILS_ENABLE_MPI -DCCUTILS_NO_JSON

all: graph500_reference_bfs_sssp graph500_reference_bfs 
#graph500_custom_bfs graph500_custom_bfs_sssp

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = main.c utils.c validate.c ../aml/aml.c
HEADERS = common.h csr_reference.h bitmap_reference.h
CCUTILS = ../ccutils/install/include 

$(CCUTILS):
	cd ../ccutils && \
	cmake -S . -B build -DCCUTILS_ENABLE_MPI=ON && \
	cmake --build build && \
	cmake --install build --prefix ./install

graph500_reference_bfs_sssp: bfs_reference.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c sssp_reference.c
	$(MPICC) $(CFLAGS) -DSSSP $(LDFLAGS) $(PREPROCESSOR_FLAGS) -o graph500_reference_bfs_sssp bfs_reference.c sssp_reference.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_reference_bfs: bfs_reference.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) $(CCUTILS) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(PREPROCESSOR_FLAGS) -I$(CCUTILS) -o graph500_reference_bfs bfs_reference.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_custom_bfs: bfs_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) $(CCUTILS) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(PREPROCESSOR_FLAGS) -I$(CCUTILS) -o graph500_custom_bfs bfs_custom.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

graph500_custom_bfs_sssp: bfs_custom.c sssp_custom.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES) csr_reference.c
	$(MPICC) $(CFLAGS) $(LDFLAGS) $(PREPROCESSOR_FLAGS) -o graph500_custom_bfs_sssp bfs_custom.c sssp_custom.c csr_reference.c $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f graph500_*

deep_clean: clean
	-rm -rf ../ccutils/install ../ccutils/build;
