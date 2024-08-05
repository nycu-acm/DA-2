import numpy as np
import matplotlib.pyplot as plt

# decay_rates = [1.1, 1.05, 1.04, 1.02]
decay_rates = [1.05]
u = 1.2
l = 0.8

def f(x, r):
    return (r**(-(x-50))) * (u - l) + l

x_values = np.linspace(50, 300, 301)

for decay_rate in decay_rates:
    y_values = f(x_values, decay_rate)
    plt.plot(x_values, y_values, label=f'decay_rate = {decay_rate}', color='g')

plt.xlabel('t')
plt.ylabel('threshold ratio')
# plt.title('Plot of f(x) for x in [0, 300]')
plt.legend()
# plt.ylim(0, 0.8)
plt.grid(True)
plt.xlim(0, 300)
# plt.savefig(f'/mnt/HDD8/max/TeachAugment_point/vis/fig/{u}_{l}-r{r}.png')
plt.savefig(f'/mnt/HDD8/max/TeachAugment_point/vis/fig/{u}_{l}.png')
# plt.show()

"""
20
[0.5231, 0.5729, 0.4417, 0.5626, 0.6174, 0.5497, 0.5889, 0.6092, 0.5879,                                                                                                                                 
        0.5627, 0.5945, 0.5502, 0.5776, 0.6177, 0.5963]
50
[0.6016, 0.6174, 0.5708, 0.6247, 0.6515, 0.6063, 0.6344, 0.6402, 0.6427,                                                                                                                                 
        0.6280, 0.6422, 0.6223, 0.6265, 0.6541, 0.6406]
95
[0.6212, 0.6485, 0.6227, 0.6501, 0.6603, 0.6424, 0.6549, 0.6593, 0.6573,                                                                                                                                 
        0.6417, 0.6555, 0.6448, 0.6506, 0.6586, 0.6473]
"""
