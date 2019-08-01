#!/bin/bash

SDTBS=${SDTBS:-src/sdtbs/sdtbs}

function usage() {
    cat <<EOF
Usage: run.sh
EOF
}

policies="rr rrf"

function run_sdtbs() {
    str=`$SDTBS $* | grep "^elapsed time:" | grep -Ewo '[[:digit:]]*\.[[:digit:]]*'`
    echo -n $str
}

function run_sdtbs_method() {
    run_sdtbs -x $*
    echo -n ' '
    run_sdtbs -p rr -s $*
    echo -n ' '
    run_sdtbs -p rrf -s $*
    echo -n ' '
    run_sdtbs -p rr $*
    echo -n ' '
    run_sdtbs -p rrf $*
    echo -n ' '
}

function run_sdtbs_policy() {
    for p in $policies
    do
	run_sdtbs_method -p $p lc:32,1,32,1,1,10000
	echo -n ' '
    done
}

echo "#direct static(rr) static(rrf) dynamic(rr) dynamic(rrf)"

run_sdtbs_method lc:32,1,32,1,1,10000

echo
