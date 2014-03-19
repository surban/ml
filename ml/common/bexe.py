import os
import glob
import sys
import subprocess

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]

for cmd_filename in glob.glob(prefix + "*/CMD"):
    with open(cmd_filename) as cmdfile:
        cmd = cmdfile.read()
    print cmd_filename, ": ", cmd
    retval = subprocess.call(cmd, shell=True)
    if retval != 0:
        print "bexe: Aborting due to batch process error."
        sys.exit(retval)
    print
    os.unlink(cmd_filename)



