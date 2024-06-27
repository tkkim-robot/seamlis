
class Env:
    def __init__(self, width=20.0, height=20.0, resolution=0.1):
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