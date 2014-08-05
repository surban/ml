import ml.common.gpu
import numpy as np
from ml.common.gpu import gather
from ml.datasets import shift_gpu


if ml.common.gpu.GPU:
    generate_data = shift_gpu.generate_data
    generate_id_data = shift_gpu.generate_id_data
else:
    from . import shift_cpu
    generate_data = shift_cpu.generate_data
    generate_id_data = shift_cpu.generate_id_data


def format_sample(smpl, width, height, threshold=0.5):
    smpl = np.reshape(gather(smpl), (height, width))
    if threshold is not None:
        out = ""
        for y in range(height):
            for x in range(width):
                if smpl[y, x] >= threshold:
                    out += "#"
                else:
                    out += "."
            out += "\n"
    else:
        out = str(smpl)
    return out