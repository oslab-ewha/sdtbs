CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @CUDA_GENCODE@ -o $@

bin_PROGRAMS = sdtbs

sdtbs_SOURCES = sdtbs.c tbs_sd.cu tbs_native.cu benchmgr.cu fedkern.cu report.cu misc.cu sdtbs.h sdtbs_cu.h \
	benchapi.cu \
	sched.cu sched_rr.cu sched_rrf.cu sched_fca.cu sched_rrm.cu \
	sched_cu.cu sched_cu_rr.cu sched_cu_rrf.cu sched_cu_fca.cu sched_cu_rrm.cu

sdtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a \
		../../benchmarks/inhouse/gma/libgma.a \
		../../benchmarks/inhouse/lma/liblma.a \
		../../benchmarks/kmeans/libkmeans.a

include $(top_srcdir)/makefile.cu
