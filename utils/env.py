import numpy as np
class Env:
    def __init__(self, width=50.0, height=5.0, resolution=0.1):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per cell
        self.x_range = (0, width)
        self.y_range = (0, height)
        self.obs_boundary = self.set_obs_boundary(width, height)
        self.obs_circle = self.set_obs_circle()
        self.obs_rectangle = self.set_obs_rectangle()
        self._discretize_map()

    def _discretize_map(self):
        self.grid_width = int(self.width / self.resolution)
        self.grid_height = int(self.height / self.resolution)

    def get_map_shape(self):
        return (self.grid_height, self.grid_width)

    def continuous_to_grid(self, x, y):
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        return grid_x, grid_y

    def grid_to_continuous(self, grid_x, grid_y):
        x = grid_x * self.resolution + self.resolution / 2
        y = grid_y * self.resolution + self.resolution / 2
        return x, y
    
    def f_to_grid(self, points):
        points = np.array(points)
        original_shape = points.shape
        
        if points.ndim == 1:
            points = points.reshape(1, -1)
        
        grid_points = (points / self.resolution).astype(int)
        
        if original_shape == (2,):
            return grid_points[0]  # Return a 1D array for a single input point
        return grid_points

    def grid_to_f(self, grid_points):
        grid_points = np.array(grid_points)
        original_shape = grid_points.shape
        
        if grid_points.ndim == 1:
            grid_points = grid_points.reshape(1, -1)
        
        points = (grid_points * self.resolution) + (self.resolution / 2)
        
        if original_shape == (2,):
            return points[0]  # Return a 1D array for a single input point
        return points

    @staticmethod
    def set_obs_boundary(width, height):  # circle
        w = width
        h = height
        linewidth = 0.1
        obs_boundary = [
            [0, 0, linewidth, h],
            [0, h, w, linewidth],
            [linewidth, 0, w, linewidth],
            [w, linewidth, linewidth, h]
        ]
        return obs_boundary

    @staticmethod
    def set_obs_rectangle():
        # obs_rectangle = [
        #     [14, 12, 8, 2],
        #     [18, 22, 8, 3],
        #     [26, 7, 2, 12],
        #     [32, 14, 10, 2]
        # ]
        obs_rectangle = []
        return obs_rectangle
    @staticmethod
    def set_obs_circle():
        obs_cir = []
        return obs_cir