import tables
import numpy as np

db_path = '../data/test_db.h5'
hdf5 = tables.open_file(db_path, 'a')
"""
trajectories = hdf5.create_group('/', 'trajectories')
path_description = {}
for i in range(2):
    path_description['x' + str(i)] = tables.Float32Col(pos = i)

path = np.array([[1, 2], [3,4 ]])
trajectory = hdf5.create_table(hdf5.root.trajectories, 'trajectory_' + str(0), path_description)
trajectory.append(path.T)
trajectory.flush()
trajectory = hdf5.create_table(hdf5.root.trajectories, 'trajectory_' + str(1), path_description)
trajectory.append(path.T)
trajectory.flush()
"""
print(hdf5.root.trajectories.trajectory_0.read().T)
