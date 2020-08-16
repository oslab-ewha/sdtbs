#include "sdtbs_cu.h"

extern void start_benchruns(void);
extern void wait_benchruns(void);

BOOL
run_native_tbs(unsigned *pticks)
{
	start_benchruns();

	init_tickcount();

	wait_benchruns();

	*pticks = get_tickcount();

	return TRUE;
}
