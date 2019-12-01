import numpy as np


class CordinateGenerator:
    def __init__(self, len_r, len_c, d_model, local_block_len=None):
        self.len_r = len_r
        self.len_c = len_c
        self.d_model = d_model
        self.local_block_len = local_block_len
        self.dict = self.init_dict()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def spatial_posenc(self, position_r, position_c):
        d_model = self.d_model

        angle_rads_r = self.get_angles(position_r, np.arange(d_model)[np.newaxis, :], d_model)

        angle_rads_c = self.get_angles(position_c, np.arange(d_model)[np.newaxis, :], d_model)

        pos_encoding = np.zeros(angle_rads_r.shape, dtype=np.float32)

        pos_encoding[:, 0::2] = np.sin(angle_rads_r[:, 0::2])

        pos_encoding[:, 1::2] = np.cos(angle_rads_c[:, 1::2])

        return pos_encoding

    def init_dict(self):
        dict = {}
        local_block_len = self.local_block_len
        if not local_block_len:
            mtx_r = np.repeat(np.arange(self.len_r)[:, np.newaxis], [self.len_c], axis=1)
            mtx_c = np.repeat(np.arange(self.len_c)[np.newaxis, :], [self.len_r], axis=0)
            for r in range(self.len_r):
                for c in range(self.len_c):
                    cor = '[{}, {}]'.format(r, c)
                    mtx_r_flat = (mtx_r - r).flatten()[:, np.newaxis]
                    mtx_c_flat = (mtx_c - c).flatten()[:, np.newaxis]
                    pos_cors = self.spatial_posenc(mtx_r_flat, mtx_c_flat)
                    dict[cor] = np.array(pos_cors, dtype=np.float32)
        else:
            full_len = 2 * local_block_len + 1
            mtx_r = np.repeat(np.arange(full_len)[:, np.newaxis], [full_len], axis=1)
            mtx_c = np.repeat(np.arange(full_len)[np.newaxis, :], [full_len], axis=0)
            cor = '[{}, {}]'.format(local_block_len, local_block_len)
            mtx_r_flat = (mtx_r - local_block_len).flatten()[:, np.newaxis]
            mtx_c_flat = (mtx_c - local_block_len).flatten()[:, np.newaxis]
            pos_cors = self.spatial_posenc(mtx_r_flat, mtx_c_flat)
            dict[cor] = np.array(pos_cors, dtype=np.float32)

        return dict

    def get(self, r, c):
        if not self.local_block_len:
            assert 0 <= r < self.len_r and 0 <= c < self.len_c
            cor = '[{}, {}]'.format(r, c)
        else:
            cor = '[{}, {}]'.format(self.local_block_len, self.local_block_len)
        return self.dict[cor]


if __name__ == "__main__":
    g = CordinateGenerator(16, 12, 64)
    mtx_dict = g.init_dict()
