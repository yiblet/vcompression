#!/bin/bash

cd "$(dirname "$0")" || (
    echo "error: couldn't cd" >&2
    exit
)

pip3 install -e compression
