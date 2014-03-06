import common.gpu

if common.gpu.GPU:
    from . import shift_gpu
    generate_data = shift_gpu.generate_data
    generate_id_data = shift_gpu.generate_id_data
else:
    from . import shift_cpu
    generate_data = shift_cpu.generate_data
    generate_id_data = shift_cpu.generate_id_data

