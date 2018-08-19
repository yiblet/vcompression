#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sets up the file for specifics
"""

import argparse
import ffmpeg

def main():
    """resizes file into expected format"""
    parser = argparse.ArgumentParser(description='Process some mpeg videos')
    parser.add_argument('input', type=str, help='input file (mp4)')
    parser.add_argument('output', type=str, help='output file (mp4)')
    args = parser.parse_args()
    (
        ffmpeg
        .input(args.input)
        .filter('fps', fps=60, round='up')
        .filter('scale', 480, -1)
        .output(args.output)
        .overwrite_output()
        .run()
    )

if __name__ == "__main__":
    main()
