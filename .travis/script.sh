#!/bin/sh
if ! [ -x "$(command -v fd)" ]; then
    alias fd=fdfind
fi

pytest --workers auto "$(fd 'test_\w*.py$')"
