.cu.o:
	@NVCC_PATH@ -c -I../include @CUDA_CPPFLAGS@ @CUDA_GENCODE@ $< -o $@