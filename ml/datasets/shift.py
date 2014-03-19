import ml.common.gpu
from ml.datasets import shift_gpu

if ml.common.gpu.GPU:
    generate_data = shift_gpu.generate_data
    generate_id_data = shift_gpu.generate_id_data
else:
    from . import shift_cpu
    generate_data = shift_cpu.generate_data
    generate_id_data = shift_cpu.generate_id_data

