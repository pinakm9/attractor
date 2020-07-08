import numpy as np
import matplotlib.pyplot as plt
import tables

# the Henon map
def henon(x, a = 1.4, b = 0.3):
    return np.array([1.0  - a*x[0]**2 + x[1], b*x[0]])

# generates a trajectory
def gen_path(func, start, length, dimension, **params):
    path = np.zeros((dimension, length))
    x = start
    for t in range(length):
        res = func(x, **params)
        path[:, t] = res
        x = res
    return path

# plots a trajectory
def plot_path(path):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(path[0, :], path[1, :], color = 'orange', s = 0.2)
    plt.show()

# generates a trajectory and goes through "burn-in" period to get to the attractor
def burn_in(func, start, length, dimension, **params):
    path = gen_path(func, start, length, dimension, **params)
    plot_path(path)
    with open('../data/burn_in.txt', 'a') as burn_in_file:
        burn_in_file.write('map: {} \t parameters: {} \t starting point: {} \t burn-in period: {} \t point on attractor: {}\n'.format(func.__name__, params, start, length, path[:, -1]))

# find a point on the attractor of the Henon map
"""
a, b = 1.4, 0.3
burn_in(func = henon, start = [(1.0-b)/2.0, (1.0-b)/2.0], length = 50000, dimension = 2, a = a, b = b)
"""

# initializes database to store attractor data
def init_database(db_path):
    hdf5 = tables.open_file(db_path, 'w')
    trajectories = hdf5.create_group('/', 'trajectories')
    hdf5.close()

# addes a new trajectory in the attractor database
def add_new_path(db_path, path_index, func, start, length, dimension, chunk_size, **params):
    class PathDescription(tables.IsDescription):
        ...

    for i in range(dimension):
        var = 'x' + str(i)
        setattr(PathDescription, var, tables.Float32Col(pos = i))
    hdf5 = tables.open_file('db_path', 'a')
    trajectory = hdf5.create_table('/trajectories', 'trajectory_' + str(path_index), PathDescription)
    row = trajectory.row()
    for i in range(length/chunk_size):
        path = gen_path(func, start, length, dimension, **params)
        for j in range(chunk_size):
            for d in range(dimension):
                row['x' + str(d)] = path[d][j]
            row.append()
        trajectory.flush()
        start = path[:, -1]
    hdf5.close()

# adds to an existing trajectory in the attractor database
def add_to_path(db_path, path_index, func, length, dimension, **params):
    hdf5 = tables.open_file('db_path', 'a')
    trajectory = getattr(hdf5.root.trajectories, 'trajectory_' + str(path_index))
    row = trajectory.row()
    start = np.array(trajectory[-1].fetch_all_field(), dtype = 'float32')
    for i in range(length/chunk_size):
        path = gen_path(func, start, length, dimension, **params)
        for j in range(chunk_size):
            for d in range(dimension):
                row['x' + str(d)] = path[d][j]
            row.append()
        trajectory.flush()
        start = path[:, -1]
    hdf5.close()

init_database('../data/henon_attractor.h5')
