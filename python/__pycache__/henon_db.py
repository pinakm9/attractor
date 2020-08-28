# create attractor database\ for the Henon map

import attract as atr

# the Henon map
def henon(x, a = 1.4, b = 0.3):
    return np.array([1.0  - a*x[0]**2 + x[1], b*x[0]])



henon_db = atr.AttractorDB(db_path = '../data/henon_attractor.h5', func = henon, dim = 2, a = 1.4, b = 0.3)
#henon_db.add_new_paths(), )
