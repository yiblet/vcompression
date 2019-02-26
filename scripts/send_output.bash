#!/bin/bash

location=$1
run_type=$2
last=$(date +"%s")
while read -r _; do
    cur=$(date +"%s")
    if [[ $((last + 30)) -le $cur ]]; then
        last=$cur
        gsutil cp -r "${location:-/content/output.txt}" "gs://yiblet_research/${run_type:-default}_output.txt"
    fi
done
