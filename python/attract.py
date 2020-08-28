# Implements class(es) for creating attractor database for a dynamical system

import numpy as np
import matplotlib.pyplot as plt
import tables
import utility as ut
import os
import tess
import scipy.spatial as ss

class AttractorDB:
    """
    Description:
        A class for creating/editing attractor database of a dynamical system

    Attrs:
        db_path: database file path
        func: dynamical function
        dim: dimension of the dynamical system
        params: dict of keyword arguments for the dynamical function
        num_paths: current number of trajectories in the database
        point_description: description of an attractor point as a hdf5 row

    Methods:
        gen_path: generates a new trajectory
        add_new_pts: addes new points to the database
        add_new_paths: adds new trajectories of equal length to the database
        add_to_path: extends an already existing trajectory in the database
        burn_in: moves a point forward in time for a long enough period for it to reach the attractor
        plot_path2D: plots an existing trajectory using only the first two coordinates
        collect_seeds: randomly collects seeds (indices of attractor points) for Voronoi tessellation and saves them in the seeds group
        tessellate: creates Voronoi tessellation from the seeds and saves it
    """

    def __init__(self, db_path, func, dim, **params):
        """
        Args:
            db_path: database file path
            func: dynamical function
            dim: dimension of the dynamical system
            **params: dict of keyword arguments for the dynamical function
        """
        # initializes database to store attractor data
        self.db_path = db_path
        self.func = func
        self.dim = dim
        self.params = params
        self.point_description = {}
        for i in range(self.dim):
            self.point_description['x' + str(i)] = tables.Float64Col(pos = i)

        if not os.path.isfile(db_path):
            hdf5 = tables.open_file(db_path, 'w')
            trajectories = hdf5.create_group('/', 'trajectories')
            points = hdf5.create_table(hdf5.root, 'points', self.point_description)
            points.flush()
            self.num_paths = 0
            hdf5.close()
        else:
            # figure out number of trajectories in a non-empty database
            hdf5 = tables.open_file(db_path, 'a')
            idx = [int(path_name.split('_')[-1]) for path_name in hdf5.root.trajectories._v_children]
            self.num_paths = max(idx) if len(idx) > 0 else 0
            hdf5.close()



    @ut.timer
    def gen_path(self, start, length):
        """
        Description:
            Generates a new trajectory

        Args:
            start: the point to start from
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectory

        Returns:
            the generated trajectory as a numpy array
        """
        path = np.zeros((self.dim, length))
        x = start
        for t in range(length):
            res = self.func(x, **self.params)
            path[:, t] = res
            x = res
        return path

    @ut.timer
    def burn_in(self, start='random', burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Description:
            Moves a point forward in time for a long enough period for it to reach the attractor

        Args:
            start: the point to start from, default = 'random' in which case a random point from a normal distribution will be selected
            burn_in_period: amount of time the point is to be moved according to the dynamics, default = 10,000
            mean: mean of the normal distribution from which the starting point is to be selected, default = None which means trajectory will start at zero vector
            cov: cov*Identity is the covarience of the normal distribution from which the starting point is to be selected, default = 0.01

        Returns:
            the final point on the attractor
        """
        if self.dim > 1:
            if mean is None:
                mean = np.zeros(self.dim)
        else:
            if mean is None:
                mean = 0.0

        def new_start():
            print('Invalid starting point, generatimng new random starting point ...')
            return np.random.multivariate_normal(mean, cov*np.eye(self.dim)) if self.dim > 1 else np.random.normal(mean, cov)
        if start is 'random':
            start = new_start()
        end_pt = self.gen_path(start, burn_in_period)[:, -1]
        while np.any(np.isinf(end_pt)):
            end_pt = self.gen_path(new_start(), burn_in_period)[:, -1]
        return end_pt


    @ut.timer
    def add_new_paths(self, num_paths=1, start='random', length=int(1e3), chunk_size=None, burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Description:
            Adds new trajectories of same length to the database

        Args:
            num_paths: number of new trajectories to be added
            start: the list of points to start from, deafult = 'random' in which case random starting points will be created via burn_in
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectories, default = int(1e3)
            chunk_size: portion of the trajectory to be wriiten to the database at a time, default = None, behaves as,
            if length < 1e4:
                chunk_size = length
            elif chunk_size is None:
                chunk_size = int(1e4)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        if length < 1e4:
            chunk_size = length
        elif chunk_size is None:
            chunk_size = int(1e4)
        if start is 'random':
            start = [self.burn_in(burn_in_period=burn_in_period, mean=mean, cov=cov) for i in range(num_paths)]
        for i in range(num_paths):
            self.num_paths += 1
            trajectory = hdf5.create_table(hdf5.root.trajectories, 'trajectory_' + str(self.num_paths), self.point_description)
            origin = start[i]
            for i in range(int(length/chunk_size)):
                path = self.gen_path(origin, chunk_size)
                trajectory.append(path.T)
                trajectory.flush()
                origin = path[:, -1]
                print('Chunk #{} has been written.'.format(i))
        hdf5.close()

    @ut.timer
    def add_to_path(self, path_index, length=int(1e3), chunk_size=None):
        """
        Description:
            Extends an already existing trajectory in the database

        Args:
            path_index: index of the path to be extended
            length: amount of time the point is to be moved according to the dynamics or the length of the trajectory, default = int(1e3)
            chunk_size: portion of the trajectory to be wriiten to the database at a time, default = None, behaves as,
            if length < 1e4:
                chunk_size = length
            elif chunk_size is None:
                chunk_size = int(1e4)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        trajectory = getattr(hdf5.root.trajectories, 'trajectory_' + str(path_index))
        if length < 1e4:
            chunk_size = length
        elif chunk_size is None:
            chunk_size = int(1e4)
        start = np.array(list(trajectory[-1]), dtype = 'float64')
        for i in range(int(length/chunk_size)):
            path = self.gen_path(start, length)
            trajectory.append(path.T)
            trajectory.flush()
            start = path[:, -1]
            print('Chunk #{} has been written.'.format(i))
        hdf5.close()


    @ut.timer
    def add_new_pts(self, num_pts=int(1e3), reset=None, burn_in_period=int(1e5), mean=None, cov=0.001):
        """
        Args:
            num_pts: number of new points to be added, default=int(1e3)
            reset: number of points before the starting point resets, default = None, behaves as,
            if num_pts < 1e3:
                reset = num_pts
            elif reset is None:
                reset = int(1e3)
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        points = hdf5.root.points
        if num_pts < 1e3:
            reset = num_pts
        elif reset is None:
            reset = int(1e3)
        for i in range(int(num_pts/reset)):
            start = self.burn_in(burn_in_period=burn_in_period, mean=mean, cov=cov)
            path = self.gen_path(start, reset)
            points.append(path.T)
            points.flush()
            print('Chunk #{} has been written.'.format(i))
        hdf5.close()

    @ut.timer
    def collect_seeds(self, num_seeds=int(1e3)):
        """
        Description:
            Randomly collects seeds (indices of attractor points) for Voronoi tessellation and saves them in the seeds group

        Args:
            num_seeds: number of seeds to be collected
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        description = {'index': tables.Int32Col(pos=0)}
        ints = np.random.choice(hdf5.root.points.shape[0], size=num_seeds, replace=False)
        try:
            hdf5.remove_node(hdf5.root.seeds)
        except:
            pass
        seeds = hdf5.create_table(hdf5.root, 'seeds', description)
        seeds.append(ints)
        seeds.flush()
        hdf5.close()

    @ut.timer
    def tessellate(self):
        """
        Description:
            Creates Voronoi tessellation from the seeds and saves it
        """
        hdf5 = tables.open_file(self.db_path, 'a')
        try:
            hdf5.remove_node(where=hdf5.root, name='Voronoi', recursive=True)
        except:
            pass
        seeds = hdf5.root.seeds
        points = hdf5.root.points
        idx = np.array(seeds.read().tolist(), dtype='int32').flatten()
        vt = tess.VoronoiTess(points[idx].tolist())
        hdf5.create_group(hdf5.root, 'Voronoi')
        for cell_id, region in enumerate(vt.tess.regions):
            if len(region) > 0 and -1 not in region:
                cell = hdf5.create_table(hdf5.root.Voronoi, 'cell_' + str(idx[cell_id]), self.point_description)
                cell.append(vt.tess.vertices[region])
                cell.flush()
        hdf5.close()
        #ss.voronoi_plot_2d(vt.tess, show_vertices=False)
        #vt.plot()
        #plt.show()

    def plot_path2D(self, path_index, show=True, saveas=None):
        """
        Description:
            Plots a trajectory in the database using only the first two coordinates

        Args:
            path_index: index of the path to be plottted
            show: boolean flag to determine if the plot is to be shown, default = True
            saveas: file path for saving the plot, default = None in which case the plot won't be saved

        Returns:
            ax of the generated plot
        """
        plt.figure(figsize = (8,8))
        ax = plt.subplot(111)
        hdf5 = tables.open_file(self.db_path, 'a')
        trajectory = np.array(getattr(hdf5.root.trajectories, 'trajectory_' + str(path_index)).read().tolist(), dtype = 'float64')
        ax.scatter(trajectory[:, 0], trajectory[:, 1], color = 'orange', s = 0.2)
        if show:
            plt.show()
        if saveas is not None:
            plt.savefig(saveas)
        return ax
