#!/usr/bin/env python
# -*- encoding=utf8
#
#   Filename:       convert
#   Author:         xiahong(xiahahaha01@gmail.com)
#   Create:         2022/10/21
#   Description:    ---
"""one line of summary

description
"""
import sys
from os import path
from pydub import AudioSegment

# files                                                                         
src = sys.argv[1]
dst = sys.argv[2]

# convert wav to mp3                                                            
sound = AudioSegment.from_mp3(src)
sound.export(dst, format="wav")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--config_path", default="./config.py")
    args = parser.parse_args()
    # main(args.config_path)

