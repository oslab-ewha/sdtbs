#!/bin/bash

function usage() {
    cat <<EOF
Usage: run.sh [options] <benchmark code:bench args only>...
 -c <# of tests>, default: 1
 -s <# of SM>, default: 1
 -p <policies>: hw,hwR,rr,rrS,rrf,rrfS,fca,rrm:64,rrmS:64
 -n <tbs per SM>, default: 1,2,3,4
 -t <threads per TB>, default: 32,64,128,256,512,1024
 -T <#TB of start>,<#TB of end>: TB incresing mode
 -x: benchmark direct mode(benchmark TB, threads should be specified) 
 -H: display header
EOF
}

SDTBS=${SDTBS:-src/sdtbs/sdtbs}
SDTBS_ARGS=${SDTBS_ARGS}

n_tests=1
n_sms=1
policies="hw,hwR,rrS,rrD,rr"
n_tbs_per_sm="1,2,3,4,5,6,7,8"
n_ths_per_tb="32,64,128,256,512,1024"
tb_range=
while getopts "c:s:p:n:t:T:xH" arg
do
    case $arg in
	c)
	    n_tests=$OPTARG
	    ;;
	s)
	    n_sms=$OPTARG
	    ;;
	p)
	    policies=$OPTARG
	    ;;
	n)
	    n_tbs_per_sm=$OPTARG
	    ;;
	t)
	    n_ths_per_tb=$OPTARG
	    ;;
	T)
	    tb_range=$OPTARG
	    ;;
	x)
	    benchdirect_mode=true
	    ;;
	H)
	    display_header=true
	    ;;
	*)
	    usage
	    exit 1
    esac
done

shift `expr $OPTIND - 1`

if [ $# -eq 0 ]; then
    usage
    exit 1
fi

function get_elapsed_time() {
    sum=0
    for i in `seq 1 $n_tests`
    do
	str=`$SDTBS $SDTBS_ARGS $* 2> /dev/null | grep "^elapsed time:" | grep -Ewo '[[:digit:]]*\.[[:digit:]]*'`
	if [ $? -ne 0 ]; then
	    echo -n "-"
	    return
	else
	    sum=`echo "scale=6;$sum + $str" | bc`
	fi
    done
    value=`echo "scale=6;$sum / $n_tests" | bc`
    echo -n $value
}

function run_sdtbs() {
    for p in $(echo $policies | tr "," "\n")
    do
	get_elapsed_time -p $p $*
	echo -n ' '
    done
    echo
}

function show_header() {
    echo -n "#"
    for p in $(echo $policies | tr "," "\n")
    do
	echo -n "$p "
    done
    echo
}

function run_sdtbs_tbs_threads() {
    for ths in $(echo $n_ths_per_tb | tr "," "\n")
    do
	cmd_args=
	for arg in $*
	do
	    bench=`echo $arg | cut -d':' -f1`
	    bencharg=`echo $arg | cut -d':' -f2`
	    cmd_args="$cmd_args $bench:$tbs,1,$ths,1,$bencharg"
	done
	run_sdtbs $cmd_args
    done
}

function run_sdtbs_tb_per_sm() {
    for m in $(echo $n_tbs_per_sm | tr "," "\n")
    do
	tbs=$(($m * $n_sms))
	run_sdtbs_tbs_threads $*
    done
}

function run_sdtbs_tbs_range() {
    for tbs in $(seq $(echo $tb_range | tr "," " "))
    do
	run_sdtbs_tbs_threads $*
    done
}

if [[ -n $display_header ]]; then
    show_header
fi

if [[ -n $benchdirect_mode ]]; then
    run_sdtbs $*
elif [[ -n $tb_range ]]; then
    run_sdtbs_tbs_range $*
else
    run_sdtbs_tb_per_sm $*
fi
