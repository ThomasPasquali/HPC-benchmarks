WITH_MPI ?= 1

CXX := mpicxx
CXXFLAGS := -O3 -std=c++17

ifndef CCUTILS_INCLUDE
$(warning CCUTILS_INCLUDE is not set!)
$(warning Please set CCUTILS_INCLUDE to the include directory of ccutils)
$(error Example: install by running the provided install.sh)
endif

CXXFLAGS += -I$(CCUTILS_INCLUDE)

# MPI handling
ifeq ($(WITH_MPI),1)
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LIBS   := $(shell mpicxx --showme:link)
CXXFLAGS  += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif

DP_PATH = ./DNNProxy/cpp/data_parallel

all: dp fsdp

dp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o dp $(DP_PATH)/dp.cpp

fsdp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o fsdp $(DP_PATH)/fsdp.cpp

clean:
	rm -f dp fsdp

.PHONY: all clean dp fsdp
