import numpy as np

class SimpleSENSOR(object):

    def __init__(self, n_sensor=7, sensor_max=2, range_sensor=[-np.pi, np.pi], noise_std=0.2):

        self.n_sensor = n_sensor
        self.sensor_max = sensor_max
        self.sensor_angle = np.linspace(range_sensor[0], range_sensor[1], n_sensor)

        self.sensor_info = sensor_max + np.zeros((n_sensor, 3))

        self.noise_std = noise_std

    def update_sensors(self, vehicle_info, obstacle_info, boundary_info):

        self.cx, self.cy, self.yaw = vehicle_info.reshape([-1]).tolist()

        self.transform_end_sensors()

        for s_idx in range(self.n_sensor):

            s = self.sensor_info[s_idx, -2:] - np.array([self.cx, self.cy])
            self.sensor_distance_check = [self.sensor_max]
            self.intersections_check = [self.sensor_info[s_idx, -2:]]

            if obstacle_info is None:

                pass
                
            else:

                for ob_idx in range(obstacle_info.shape[0]):

                    self.check_obs_cast(s, obstacle_info[ob_idx, :, :])

            self.check_bound_cast(s, boundary_info)

            distance = np.min(self.sensor_distance_check)
            distance_index = np.argmin(self.sensor_distance_check)
            self.sensor_info[s_idx, 0] = np.clip(distance + self.noise_std * np.random.randn(1), 0, self.sensor_max)  
            self.sensor_info[s_idx, -2:] = self.intersections_check[distance_index]
            

    def transform_end_sensors(self):
        
        xs = self.sensor_max * np.ones((self.n_sensor, )) * np.cos(self.sensor_angle)
        ys = self.sensor_max * np.ones((self.n_sensor, )) * np.sin(self.sensor_angle)
        xys = np.concatenate([xs.reshape([-1,1]), ys.reshape([-1,1])], axis=1)

        xs_rot = xs * np.cos(self.yaw) - ys * np.sin(self.yaw)
        ys_rot = xs * np.sin(self.yaw) + ys * np.cos(self.yaw)

        self.sensor_info[:, 1] = xs_rot + self.cx
        self.sensor_info[:, 2] = ys_rot + self.cy


    def check_obs_cast(self, s, obstacle_info):
        
        for oi in range(obstacle_info.shape[0]):
            p = obstacle_info[oi]
            r = obstacle_info[(oi + 1) % obstacle_info.shape[0]] - obstacle_info[oi]
            if np.cross(r, s) != 0:
                t = np.cross((np.array([self.cx, self.cy]) - p), s) / np.cross(r, s)
                u = np.cross((np.array([self.cx, self.cy]) - p), r) / np.cross(r, s)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = np.array([self.cx, self.cy]) + u * s
                    self.intersections_check.append(intersection)
                    self.sensor_distance_check.append(np.linalg.norm(u*s))
        
        
    def check_bound_cast(self, s, boundary_info):

        for oi in range(4):
            p = boundary_info[oi]
            r = boundary_info[(oi + 1) % boundary_info.shape[0]] - boundary_info[oi]
            if np.cross(r, s) != 0:  # may collision
                t = np.cross((np.array([self.cx, self.cy]) - p), s) / np.cross(r, s)
                u = np.cross((np.array([self.cx, self.cy]) - p), r) / np.cross(r, s)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    intersection = p + t * r
                    self.intersections_check.append(intersection)
                    self.sensor_distance_check.append(np.linalg.norm(intersection - np.array([self.cx, self.cy])))

