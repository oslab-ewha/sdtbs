CLEANFILES = *~
DISTCLEANFILES = *~

LINK = @NVCC_PATH@ @NVCC_ARCHITECTURE@ -o $@

bin_PROGRAMS = sdtbs

sdtbs_SOURCES = sdtbs.c tbs_sd.cu tbs_native.cu benchmgr.cu benchrun.cu skrun.cu \
	fedkern.cu fedkern_dyn.cu fedkern_host.cu \
	report.cu misc.cu sdtbs.h sdtbs_cu.h benchapi.cu \
	sched.cu sched_rr.cu sched_rrf.cu sched_fca.cu sched_rrm.cu \
	sched_host.cu sched_prl.cu \
	sched_dyn.cu sched_dyn_rr.cu sched_dyn_rrf.cu sched_dyn_fca.cu sched_dyn_rrm.cu \
	sched_prl_rrp.cu

sdtbs_LDADD = ../../benchmarks/inhouse/loopcalc/libloopcalc.a \
		../../benchmarks/inhouse/gma/libgma.a \
		../../benchmarks/inhouse/lma/liblma.a \
		../../benchmarks/kmeans/libkmeans.a

include $(top_srcdir)/makefile.cu
