# Creates and modifies attractor database for henon map
import attract as atr
import numpy as np
import matplotlib.pyplot as plt

# the Henon map
def henon(x, a, b):
    return np.array([1.0  - a*x[0]**2 + x[1], b*x[0]])

henon_db = atr.AttractorDB(db_path = '../data/henon_attractor.h5', func = henon, dim = 2, a = 1.4, b = 0.3)
#henon_db.add_new_paths(num_paths = 1, length = 1000)
#henon_db.plot_path2D(9)

#henon_db.add_new_pts(10000)
henon_db.collect_seeds(num_seeds = 50)
henon_db.tessellate(image_path = '../images/henon_14_3_Vor_50.png')
#henon_db.assign_pts_to_cells()
"""
sampler = atr.AttractorSampler(db_path = '../data/henon_attractor.h5')
pts = np.random.normal(size=(4, 2))
print(sampler.resample(pts))
"""
sampler = atr.AttractorSampler(db_path = '../data/henon_attractor.h5')
plt.figure(figsize = (8,8))
ax = plt.subplot(111)
trajectory = sampler.points[1000:2000]
ax.scatter(trajectory[:, 0], trajectory[:, 1], color = 'orange', s = 0.2)
plt.show()
