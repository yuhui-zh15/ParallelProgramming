CC			:= gcc
CXX			:= g++

MPICC			:= mpicc
MPICXX			:= mpicxx

CCFLAGS			:= -O3 -march=native -std=gnu99 
CXXFLAGS		:= -O3 -march=native -std=gnu++03
LDFLAGS			:= -lX11 -fopenmp -lm 

all: MS_hybrid MS_hybrid_static

MS_hybrid: HW2_x1054034.c
	$(MPICC) $(CCFLAGS) $(LDFLAGS) -o $@ $?
MS_hybrid_static: HW2_x1054034_static.c
	$(MPICC) $(CCFLAGS) $(LDFLAGS) -o $@ $?
clean:
	rm -f MS_hybrid MS_hybrid_static
submit:
	qsub job.sh
