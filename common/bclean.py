import os
import glob
import sys
import subprocess

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]

for cmd_filename in glob.glob(prefix + "*/CMD"):
    print cmd_filename
    os.unlink(cmd_filename)



