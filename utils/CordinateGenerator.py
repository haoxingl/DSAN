import numpy as np


class CordinateGenerator:
    def __init__(self, len_r, len_c):
        self.len_r = len_r
        self.len_c = len_c
        self.dict = self.init_dict()

    def init_dict(self):
        dict = {}
        mtx_r = np.repeat(np.arange(self.len_r)[:, np.newaxis], [self.len_c], axis=1)
        mtx_c = np.repeat(np.arange(self.len_c)[np.newaxis, :], [self.len_r], axis=0)
        for r in range(self.len_r):
            for c in range(self.len_c):
                cor = '[{}, {}]'.format(r, c)
                mtx_r_flat = (mtx_r - r).flatten()[:, np.newaxis]
                mtx_c_flat = (mtx_c - c).flatten()[:, np.newaxis]
                pos_cors = np.concatenate([mtx_r_flat, mtx_c_flat], axis=-1)
                dict[cor] = np.array(pos_cors, dtype=np.int64)

        return dict

    def get(self, r, c):
        assert 0 <= r < self.len_r and 0 <= c < self.len_c
        cor = '[{}, {}]'.format(r, c)
        return self.dict[cor]


if __name__ == "__main__":
    g = CordinateGenerator(16, 12)
    mtx_dict = g.init_dict()
