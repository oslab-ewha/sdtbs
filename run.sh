#!/bin/bash

function usage() {
    cat <<EOF
Usage: run.sh [options] <benchmark code:bench args only>...
 -s <# of SM>, default: 1
EOF
}

SDTBS=${SDTBS:-src/sdtbs/sdtbs}
SDTBS_ARGS=${SDTBS_ARGS}

n_sms=1
while getopts "s:" arg
do
    case $arg in
	s)
	    n_sms=$OPTARG
	;;
    esac
done

shift `expr $OPTIND - 1`

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

function get_elapsed_time() {
    str=`$SDTBS $SDTBS_ARGS $* 2> /dev/null | grep "^elapsed time:" | grep -Ewo '[[:digit:]]*\.[[:digit:]]*'`
    if [ $? -ne 0 ]; then
	echo -n "-"
    else
	echo -n $str
    fi
}

function run_sdtbs() {
    get_elapsed_time -x $*
    echo -n ' '
    get_elapsed_time -p rr -s $*
    echo -n ' '
    get_elapsed_time -p rrf -s $*
    echo -n ' '
    get_elapsed_time -p rr $*
    echo -n ' '
    get_elapsed_time -p rrf $*
    echo
}

function run_sdtbs_tbs_threads() {
    for m in `seq 1 4`
    do
	tbs=$(($m * $n_sms))
	for ths in 32 64 128 256 512 1024
	do
	    for arg in $*
	    do
		bench=`echo $arg | cut -d':' -f1`
		bencharg=`echo $arg | cut -d':' -f2`
		run_sdtbs $bench:$tbs,1,$ths,1,$bencharg
	    done
	done
    done
}

echo "#direct static(rr) static(rrf) dynamic(rr) dynamic(rrf)"

run_sdtbs_tbs_threads $*
