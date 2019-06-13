CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @CUDA_GENCODE@ -o $@

bin_PROGRAMS = sdtbs

sdtbs_SOURCES = sdtbs.c macrotb.cu misc.cu

sdtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a

include $(top_srcdir)/makefile.cu
