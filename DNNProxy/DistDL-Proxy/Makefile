MPICXX = mpicxx

SRC_PATH = ./proxies
CCUTILS_PATH = $(HOME)/ccutils/include

CXXFLAGS = -O3 -I$(CCUTILS_PATH) $(shell mpicxx --showme:compile) -DCCUTILS_ENABLE_MPI
LDLIBS   += $(shell mpicxx --showme:link)


all: resnet152

resnet152:
	$(MPICXX) $(CXXFLAGS) -o $(SRC_PATH)/$@ $(SRC_PATH)/$@.cpp 

clean:
	rm $(SRC_PATH)/resnet152