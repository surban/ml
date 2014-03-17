#!/bin/bash

device=$1

if [ "$device" == "cpu32" ] ; then
    export GNUMPY_CPU_PRECISION=32
    export GNUMPY_USE_GPU=no
    export THEANO_FLAGS=floatX=float32,device=cpu
elif [ "$device" == "cpu64" ] ; then
    export GNUMPY_CPU_PRECISION=64
    export GNUMPY_USE_GPU=no
    export THEANO_FLAGS=floatX=float64,device=cpu
elif [ "$device" == "gpu" ] ; then
    export GNUMPY_CPU_PRECISION=32
    export GNUMPY_USE_GPU=yes
    export THEANO_FLAGS=floatX=float32,device=gpu
else 
    echo "Unknown device: $device"
fi

