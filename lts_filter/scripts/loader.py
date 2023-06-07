#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, data, lidar: str = 'vlp-16') -> None:
        super().__init__()

        assert lidar in {'vlp-16'}, 'lidar type should be \'vlp-16\''

        lidar_params = {
            'vlp-16': {
                'num_beams': 16,
                'fov_up': 16.8,
                'fov_down': -16.8,
                'window_size': 128,
            },
        }
        lidar_param = lidar_params[lidar]

        self.num_slices = 512
        self.num_windows = 1
        self.frame = self.lidar_to_image(data, lidar_param, self.num_slices)


    def lidar_to_image(self, data: np.ndarray, lidar_param: dict, num_slices: int) -> np.ndarray:
        data = np.unique(data, axis=0)
        data = data[data[:, 3] != -1]

        x, y, z, s = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        theta = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
        phi = np.arctan2(y, x) * 180 / np.pi

        fov_total = lidar_param['fov_up'] - lidar_param['fov_down']
        theta_res = fov_total / (lidar_param['num_beams'] - 1)
        phi_res = 360 / num_slices

        theta_idx = np.floor((theta - lidar_param['fov_down']) / theta_res).astype(np.int32)
        phi_idx = np.floor(phi / phi_res).astype(np.int32)

        num_channels = 4
        projected_data = np.zeros((lidar_param['num_beams'], num_slices, num_channels), dtype=np.float32)

        projected_data[theta_idx, phi_idx, 0] = x
        projected_data[theta_idx, phi_idx, 1] = y
        projected_data[theta_idx, phi_idx, 2] = z
        projected_data[theta_idx, phi_idx, 3] = s

        return projected_data


    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        frame = self.frame[:, :, :].reshape(-1, self.frame.shape[-1])

        points = frame[:, :3]
        labels = frame[:, 3]

        return points, labels

    def __len__(self) -> int:
        return self.num_windows
