
CXX=g++


ifeq (unix,macos)
  LIB_FLAGS = -larmadillo -framework Accelerate
else
  LIB_FLAGS = -larmadillo -llapack -lblas -DARMA_DONT_USE_WRAPPER 
endif


OPT = -O2 -mcmodel=medium  -fopenmp

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: SNeCT


SNeCT: SNeCT.cpp 
	$(CXX) $(CXXFLAGS)  -o $@  $< $(LIB_FLAGS)


.PHONY: clean

clean:
	rm -f SNeCT