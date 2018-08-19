#!/bin/sh
pytest --workers auto
find src -iname "*.py" | xargs pylint -j 0
find scripts -iname "*.py" | xargs pylint -j 0
