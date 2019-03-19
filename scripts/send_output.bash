#!/bin/bash

location=$1
run_type=$2
last=$(date +"%s")
while read -r _; do
    cur=$(date +"%s")
    if [[ $((last + 30)) -le $cur ]]; then
        last=$cur
        gsutil cp -r "${location:-/content/output.txt}" "gs://yiblet_research/logs/${run_type:-default}.txt"
    fi
done
