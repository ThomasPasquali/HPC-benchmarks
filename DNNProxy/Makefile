WITH_MPI   ?= 1
PROXY_CUDA ?= 0
PROXY_HIP  ?= 0
HALF       ?= 0

CXX := mpicxx
CXXFLAGS := -O3 -std=c++17

ifndef CCUTILS_INCLUDE
$(warning CCUTILS_INCLUDE is not set!)
$(warning Please set CCUTILS_INCLUDE to the include directory of ccutils)
$(error Example: install by running the provided install.sh)
endif

# CUDA / HIP selection
ifeq ($(PROXY_CUDA),1)
    ifeq ($(PROXY_HIP),1)
        $(error Cannot enable both PROXY_CUDA and PROXY_HIP)
    endif
    CXX := nvcc
    CXXFLAGS += -DPROXY_ENABLE_CUDA -DCCUTILS_ENABLE_CUDA
endif

ifeq ($(PROXY_HIP),1)
    CXX := hipcc
    CXXFLAGS += -DPROXY_ENABLE_HIP -DCCUTILS_ENABLE_HIP
endif

# HALF precision requires CUDA or HIP
ifeq ($(HALF),1)
    ifneq ($(PROXY_CUDA),1)
        ifneq ($(PROXY_HIP),1)
            $(error HALF requires PROXY_CUDA or PROXY_HIP)
        endif
    endif
    CXXFLAGS += -DHALF_PRECISION
endif

CXXFLAGS += -I$(CCUTILS_INCLUDE)

# MPI handling
ifeq ($(WITH_MPI),1)
export OMPI_MCA_opal_cuda_support=1
MPI_CFLAGS := $(shell mpicxx --showme:compile)
MPI_LIBS   := $(shell mpicxx --showme:link)

ifeq ($(PROXY_CUDA),1)
    CXXFLAGS += $(filter-out -pthread,$(MPI_CFLAGS)) -DWITH_MPI -DCCUTILS_ENABLE_MPI
    MPI_LIBS := $(filter-out -pthread,$(MPI_LIBS))
else
    CXXFLAGS += $(MPI_CFLAGS) -DWITH_MPI -DCCUTILS_ENABLE_MPI
endif
endif

DP_PATH = ./DNNProxy/cpp/data_parallel

all: dp fsdp

dp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o dp $(DP_PATH)/dp.cpp

fsdp:
	$(CXX) $(CXXFLAGS) $(MPI_LIBS) -o fsdp $(DP_PATH)/fsdp.cpp

clean:
	rm -f dp fsdp