import numpy as np

from safe_control.tracking import LocalTrackingController as _BaseLocalTrackingController


class LocalTrackingController(_BaseLocalTrackingController):
    def update_goal(self):
        if self.robot_spec['model'] in ['Quad3D']:
            n_pos = 3
        else:
            n_pos = 2

        if self.state_machine == 'rotate':
            current_angle = self.robot.get_orientation()
            goal_angle = np.arctan2(self.waypoints[0][1] - self.robot.X[1, 0],
                                    self.waypoints[0][0] - self.robot.X[0, 0])
            if self.robot_spec['model'] in ['Quad2D', 'VTOL2D', 'Manipulator2D']:
                self.state_machine = 'track'
            if not self.enable_rotation:
                self.state_machine = 'track'
            if abs(current_angle - goal_angle) > self.rotation_threshold:
                return self.waypoints[0][:n_pos]
            else:
                self.state_machine = 'track'
                self.u_att = None

        if self.current_goal_index >= len(self.waypoints):
            return None

        if self.goal_reached(self.robot.X, np.array(self.waypoints[self.current_goal_index]).reshape(-1, 1)):
            self.current_goal_index += 1
            if self.current_goal_index >= len(self.waypoints):
                self.state_machine = 'idle'
                return None

        goal = np.array(self.waypoints[self.current_goal_index][0:n_pos])
        return goal

    def set_waypoints(self, waypoints):
        if type(waypoints) == list:
            waypoints = np.array(waypoints, dtype=float)
        waypoints = np.array(waypoints, dtype=float)
        if waypoints.ndim == 1 and waypoints.size > 0:
            waypoints = waypoints.reshape(1, -1)

        filtered_waypoints = self.filter_waypoints(waypoints)
        # Guard against exploration deadlock: filtering can remove all
        # waypoints when spacing is below reached_threshold.
        if filtered_waypoints.size == 0 and waypoints.size > 0:
            if waypoints.ndim == 2 and waypoints.shape[1] >= 2:
                robot_pos = self.robot.get_position()
                distances = np.linalg.norm(waypoints[:, :2] - robot_pos.reshape(1, -1), axis=1)
                keep_idx = int(np.argmax(distances))
                filtered_waypoints = waypoints[keep_idx:keep_idx + 1]
            else:
                filtered_waypoints = waypoints[-1:].copy()

        self.waypoints = filtered_waypoints
        self.current_goal_index = 0

        self.goal = self.update_goal()
        if self.goal is not None:
            if not self.robot.is_in_fov(self.goal):
                if self.robot_spec['exploration']:
                    self.state_machine = 'rotate'
                else:
                    self.state_machine = 'stop'
                    self.goal = None
            else:
                self.state_machine = 'track'

        if self.show_animation:
            self.waypoints_scatter.set_offsets(self.waypoints[:, :2])

    def get_nearest_unpassed_obs(self, detected_obs, angle_unpassed=np.pi * 2, obs_num=5):
        def angle_normalize(x):
            return (((x + np.pi) % (2 * np.pi)) - np.pi)

        if self.robot_spec['model'] in ['SingleIntegrator2D', 'DoubleIntegrator2D', 'Quad2D', 'Quad3D']:
            angle_unpassed = np.pi * 2
        elif self.robot_spec['model'] in ['Unicycle2D', 'DynamicUnicycle2D', 'VTOL2D']:
            angle_unpassed = np.pi * 1.2
        elif 'KinematicBicycle2D' in self.robot_spec['model']:
            angle_unpassed = np.pi * 2.0

        detected_arr = np.array([], dtype=float)
        if detected_obs is not None and len(detected_obs) != 0:
            detected_arr = np.array(detected_obs, dtype=float)
            if detected_arr.ndim == 1:
                detected_arr = detected_arr.reshape(1, -1)
            # Slightly inflate detected-unknown radius inside the CBF layer to
            # counter discretization / solver tolerance effects.
            unknown_margin = float(
                self.robot_spec.get(
                    'unknown_obs_cbf_margin',
                    max(0.08, 0.5 * float(self.robot.robot_radius)),
                )
            )
            detected_arr = detected_arr.copy()
            detected_arr[:, 2] = detected_arr[:, 2] + unknown_margin

        if detected_arr.size != 0:
            if len(self.obs) == 0:
                all_obs = detected_arr
            else:
                all_obs = np.vstack((self.obs, detected_arr))
        else:
            all_obs = self.obs

        if len(all_obs) == 0:
            return None

        if all_obs.ndim == 1:
            all_obs = all_obs.reshape(1, -1)

        # Inflate obstacle size used by CBF constraints to absorb solver and
        # discretization errors near obstacle boundaries.
        obs_margin = float(self.robot_spec.get('known_obs_cbf_margin', 0.08))
        if obs_margin > 0.0:
            all_obs = all_obs.copy()
            shape_flag = all_obs[:, 6] if all_obs.shape[1] > 6 else np.zeros(all_obs.shape[0])
            is_super = shape_flag > 0.5
            # Circle-like obstacles: radius field a/r at column 2.
            all_obs[~is_super, 2] = all_obs[~is_super, 2] + obs_margin
            # Superellipse obstacles: inflate both semi-axes a and b.
            if np.any(is_super):
                all_obs[is_super, 2] = all_obs[is_super, 2] + obs_margin
                if all_obs.shape[1] > 3:
                    all_obs[is_super, 3] = all_obs[is_super, 3] + obs_margin

        robot_pos = self.robot.get_position()
        robot_yaw = self.robot.get_orientation()
        to_obs_vectors = all_obs[:, :2] - robot_pos.reshape(1, -1)
        distances = np.linalg.norm(to_obs_vectors, axis=1)
        radii = all_obs[:, 2]
        clearances = distances - (radii + float(self.robot.robot_radius))

        angle_to_obs = np.arctan2(to_obs_vectors[:, 1], to_obs_vectors[:, 0])
        angle_diff = np.abs(np.array([angle_normalize(a - robot_yaw) for a in angle_to_obs]))
        unpassed_mask = angle_diff <= (angle_unpassed / 2.0)

        # Always keep very close obstacles regardless of heading to avoid
        # collisions with already-detected side/rear obstacles.
        v_max = float(self.robot_spec.get('v_max', 1.0))
        critical_clearance = float(self.robot_spec.get('critical_obs_clearance', max(0.2, 0.55 * v_max * self.dt)))
        critical_mask = clearances <= critical_clearance

        candidate_mask = np.logical_or(unpassed_mask, critical_mask)
        if np.any(candidate_mask):
            candidate_indices = np.where(candidate_mask)[0]
        else:
            candidate_indices = np.arange(all_obs.shape[0], dtype=int)

        # Slightly prioritize unknown-detected obstacles when distances are similar.
        unknown_mask = np.zeros(all_obs.shape[0], dtype=bool)
        if detected_arr.size != 0:
            unknown_mask[-detected_arr.shape[0]:] = True

        ordered_candidates = candidate_indices[np.argsort(clearances[candidate_indices])]
        nearest_indices = ordered_candidates[:obs_num].astype(int, copy=True)

        unknown_candidates = candidate_indices[unknown_mask[candidate_indices]]
        ordered_unknown = unknown_candidates[np.argsort(clearances[unknown_candidates])]
        unknown_priority_count = int(
            self.robot_spec.get(
                'unknown_obs_priority_count',
                min(4, max(2, int(max(obs_num, 1) // 6))),
            )
        )
        unknown_quota = min(obs_num, max(0, unknown_priority_count))
        if unknown_quota > 0 and ordered_unknown.size > 0 and nearest_indices.size > 0:
            selected_set = set(int(i) for i in nearest_indices.tolist())
            unknown_selected = int(np.sum(unknown_mask[nearest_indices]))
            missing_unknown = max(0, unknown_quota - unknown_selected)

            if missing_unknown > 0:
                unknown_pool = [int(i) for i in ordered_unknown.tolist() if int(i) not in selected_set]
                for unknown_idx in unknown_pool:
                    if missing_unknown <= 0:
                        break
                    non_unknown_positions = [
                        pos for pos, idx in enumerate(nearest_indices.tolist())
                        if not unknown_mask[int(idx)]
                    ]
                    if len(non_unknown_positions) == 0:
                        break
                    replace_pos = max(
                        non_unknown_positions,
                        key=lambda p: clearances[int(nearest_indices[p])],
                    )
                    nearest_indices[replace_pos] = int(unknown_idx)
                    selected_set.add(int(unknown_idx))
                    missing_unknown -= 1

            nearest_indices = nearest_indices[np.argsort(clearances[nearest_indices])]
        return all_obs[nearest_indices]
