CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @CUDA_GENCODE@ -o $@

bin_PROGRAMS = sdtbs

sdtbs_SOURCES = sdtbs.c tbs_sd.cu tbs_native.cu benchmgr.cu micro_tb.cu sched_rr.cu misc.cu

sdtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a \
		../../benchmarks/inhouse/gma/libgma.a

include $(top_srcdir)/makefile.cu
