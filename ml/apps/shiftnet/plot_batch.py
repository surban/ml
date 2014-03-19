import numpy as np
import matplotlib.pyplot as plt
import glob
import os

data = {}

for dir in glob.glob("batch_*"):
    glbs = {}
    lcls = {}
    execfile(dir + "/cfg.py", glbs, lcls)
    results = np.load(dir + "/result.npz")
    data[lcls['x_len']] = results

x_lens = sorted(data.keys())

# plot loss for various x_lens
plt.figure()
plt.hold(True)
plt.yscale('log')
for x_len in x_lens:
    results = data[x_len]
    history = results['history']
    plt.plot(history[0], history[3], label=x_len)
plt.legend()
plt.savefig("progress.pdf")

# plot x_len vs final loss
plt.figure()
plt.yscale('log')
losses = []
for x_len in x_lens:
    results = data[x_len]
    losses.append(results['global_loss'])
plt.plot(x_lens, losses, 'x-')
plt.xlabel("x_len")
plt.ylabel("final loss")
plt.savefig("final_loss.pdf")
