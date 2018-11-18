#!/bin/sh
pytest --workers auto "$(git grep -l 'test' | grep 'test_.*py$')"
