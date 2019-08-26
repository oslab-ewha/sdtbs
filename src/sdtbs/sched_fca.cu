#include "sdtbs_cu.h"

/* static scheduling for fca not supported */
sched_t	sched_fca = {
	"fca",
	FALSE, FALSE, FALSE, FALSE,
	NULL,
	NULL
};
