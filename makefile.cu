.cu.o:
	@NVCC_PATH@ -dc $(AM_CPPFLAGS) --maxrregcount 32 @CUDA_CPPFLAGS@ @CUDA_GENCODE@ $< -o $@