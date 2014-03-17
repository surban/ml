Param(
    [Parameter(Mandatory=$True,Position=1)]
    [string]$device)

if ($device -eq "cpu32") {
    $env:GNUMPY_CPU_PRECISION = "32"
    $env:GNUMPY_USE_GPU = "no"
    $env:THEANO_FLAGS = "floatX=float32,device=cpu"
} elseif ($device -eq "cpu64") {
    $env:GNUMPY_CPU_PRECISION = "64"
    $env:GNUMPY_USE_GPU = "no"
    $env:THEANO_FLAGS = "floatX=float64,device=cpu"
}  elseif ($device -eq "gpu") {
    $env:GNUMPY_CPU_PRECISION = "32"
    $env:GNUMPY_USE_GPU = "yes"
    $env:THEANO_FLAGS = "floatX=float32,device=gpu"
} else {
    echo "Unknown device: $device"
}

