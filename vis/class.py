import numpy as np
import matplotlib.pyplot as plt

classes = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]
scores = [0.6016, 0.6174, 0.5708, 0.6247, 0.6515, 0.6063, 0.6344, 0.6402, 0.6427,                                                                                                                                 
    0.6280, 0.6422, 0.6223, 0.6265, 0.6541, 0.6406]

plt.barh(classes, scores, color='#1f77b4')
# plt.ylabel('Classes')
plt.xlabel('Average True Scores')


# plt.savefig(f'/mnt/HDD8/max/TeachAugment_point/vis/fig/{u}_{l}-r{r}.png')
plt.savefig(f'/mnt/HDD8/max/TeachAugment_point/vis/fig/avg.png')
# plt.show()