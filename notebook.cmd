@echo off
call setpath.cmd
pushd notebooks
ipython notebook --pylab=inline
popd
