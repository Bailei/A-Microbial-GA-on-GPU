main: ga_g.cu
	nvcc -g -G -arch=sm_20 ga_g.cu -o main
