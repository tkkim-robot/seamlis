import numpy as np
class Env:
    def __init__(self, x_range=(-3.5, 1.0), y_range=(-2.3, 1.8), resolution=0.1):
        self.x_range = x_range
        self.y_range = y_range
        self.width = x_range[1] - x_range[0]
        self.height = y_range[1] - y_range[0]
        self.resolution = resolution  # meters per cell
        self.obs_boundary = self.set_obs_boundary()
        self.obs_circle = self.set_obs_circle()
        self.obs_rectangle = self.set_obs_rectangle()
        self._discretize_map()

    def _discretize_map(self):
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)

    def get_map_shape(self):
        return (self.grid_height, self.grid_width)
    
    def f_to_grid(self, points):
        points = np.array(points)
        original_shape = points.shape

        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Shift points to be relative to the origin of the grid
        shifted_points = points - np.array([self.x_range[0], self.y_range[0]])
        grid_points = (shifted_points / self.resolution).astype(int)

        if original_shape == (2,):
            return grid_points[0]  # Return a 1D array for a single input point
        return grid_points

    def grid_to_f(self, grid_points):
        grid_points = np.array(grid_points)
        original_shape = grid_points.shape

        if grid_points.ndim == 1:
            grid_points = grid_points.reshape(1, -1)

        # Convert grid points to continuous space and shift back to original coordinate system
        points = (grid_points * self.resolution) + (self.resolution / 2)
        points = points + np.array([self.x_range[0], self.y_range[0]])

        if original_shape == (2,):
            return points[0]  # Return a 1D array for a single input point
        return points

    def set_obs_boundary(self):  # circle
        w = self.width
        h = self.height
        linewidth = 0.1
        x_start = self.x_range[0]
        y_start = self.y_range[0]
        obs_boundary = [
            [x_start, y_start, linewidth, h],
            [x_start, y_start + h, w, linewidth],
            [x_start + linewidth, y_start, w, linewidth],
            [x_start + w, y_start + linewidth, linewidth, h]
        ]
        return obs_boundary

    @staticmethod
    def set_obs_rectangle():
        # Example obstacles in the new coordinate system
        obs_rectangle = []
        return obs_rectangle

    @staticmethod
    def set_obs_circle():
        obs_cir = []
        return obs_cir