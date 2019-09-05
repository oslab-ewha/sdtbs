#include "sdtbs_cu.h"

/* static scheduling for fca not supported */
sched_t	sched_fca = {
	"fca",
	TBS_TYPE_DYNAMIC,
	NULL,
	NULL
};
