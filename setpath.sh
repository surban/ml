#!/bin/bash

pushd $(dirname ${BASH_SOURCE[0]}) > /dev/null
export PYTHONPATH=$PYTHONPATH:$(pwd)
popd > /dev/null


