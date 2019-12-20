NVCC = nvcc
CFLAGS = -g -G -O0 -lcurand
calc_pi: kernel.cu main.h
	$(NVCC) $(CFLAGS) $< -o $@