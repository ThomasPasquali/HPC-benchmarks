CXX ?= g++
CFLAGS ?= -O3
LDLIBS ?= -pthread -std=c++11

SRC_DIR ?= src
BIN_DIR ?= bin

TARGET_NAMES := $(filter-out main, $(basename $(notdir $(wildcard $(SRC_DIR)/*.cpp))))
ALL_TARGETS := $(addprefix $(BIN_DIR)/, $(TARGET_NAMES))

all: $(ALL_TARGETS)

clean:
	rm -f $(ALL_TARGETS)

# Each binary is built from main.cpp + specific source (e.g., atomic.cpp)
$(BIN_DIR)/%: $(SRC_DIR)/main.cpp $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CFLAGS) $(LDLIBS) -o $@ $^
