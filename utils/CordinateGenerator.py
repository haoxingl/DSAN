import numpy as np


class CordinateGenerator:
    def __init__(self, len_x, len_y):
        self.len_x = len_x
        self.len_y = len_y
        self.dict = self.init_dict()

    def init_dict(self):
        dict = {}
        mtx_x = np.repeat(np.arange(self.len_x)[np.newaxis, :], [self.len_y], axis=0)
        mtx_y = np.repeat(np.arange(self.len_y)[:, np.newaxis], [self.len_x], axis=1)
        for i in range(self.len_x):
            for j in range(self.len_y):
                cor = '[{}, {}]'.format(i, j)
                mtx_x_flat = (mtx_x - i).flatten()[:, np.newaxis]
                mtx_y_flat = (mtx_y - j).flatten()[:, np.newaxis]
                pos_cors = np.concatenate([mtx_x_flat, mtx_y_flat], axis=-1)
                dict[cor] = np.array(pos_cors, dtype=np.int64)

        return dict

    def get(self, x, y):
        assert 0 <= x < self.len_x and 0 <= y < self.len_y
        cor = '[{}, {}]'.format(x, y)
        return self.dict[cor]


if __name__ == "__main__":
    g = CordinateGenerator(12, 16, 64)
    mtx_dict = g.init_dict()
