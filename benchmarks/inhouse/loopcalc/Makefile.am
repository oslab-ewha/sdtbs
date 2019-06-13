CLEANFILES = *~
DISTCLEANFILES = *~

noinst_LIBRARIES = libloopcalc.a

libloopcalc_a_SOURCES = loopcalc.cu

.cu.o:
	@NVCC_PATH@ -c -I../include @CUDA_CPPFLAGS@ @CUDA_GENCODE@ $< -o $@
