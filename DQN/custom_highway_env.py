import numpy as np
from highway_env.envs.highway_env import HighwayEnv


class CustomHighwayEnv(HighwayEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "collision_reward": -10.0,
                "relative_speed_bonus": 2.0,
                "desired_speed_margin": 2.5,
                "speed_tolerance": 2.0,
                "lane_change_penalty": 0.18,
                "left_lane_penalty": 0.04,
                "overtake_bonus": 0.25,
                "acceleration_penalty_weight": 0.10,
                "jerk_penalty_weight": 0.24,
                "action_acceleration_penalty": 0.03,
                "stable_speed_bonus": 0.10,
            }
        )
        return config

    def _reset(self) -> None:
        super()._reset()
        self.prev_speed = self.vehicle.speed
        self.prev_acceleration = 0.0
        self.prev_front_x = None

    def _get_front_vehicle(self):
        ego = self.vehicle
        same_lane_vehicles = []

        for veh in self.road.vehicles:
            if veh is ego:
                continue

            # Même voie
            if veh.lane_index == ego.lane_index:
                dx = veh.position[0] - ego.position[0]
                if dx > 0:
                    same_lane_vehicles.append((dx, veh))

        if not same_lane_vehicles:
            return None

        same_lane_vehicles.sort(key=lambda x: x[0])
        return same_lane_vehicles[0][1]

    def _reward(self, action: int) -> float:
        collision_term = self.config["collision_reward"] if self.vehicle.crashed else 0.0

        dt = 1.0 / self.config["policy_frequency"]
        speed = self.vehicle.speed
        acceleration = (speed - self.prev_speed) / dt
        jerk = (acceleration - self.prev_acceleration) / dt

        front_vehicle = self._get_front_vehicle()

        # 1) Cible de vitesse relative au véhicule de devant
        if front_vehicle is not None:
            front_speed = front_vehicle.speed
            desired_speed = front_speed + self.config["desired_speed_margin"]
        else:
            desired_speed = speed  # pas de véhicule devant -> pas de sur-incitation

        speed_error = abs(speed - desired_speed)
        tol = self.config["speed_tolerance"]
        relative_speed_score = max(0.0, 1.0 - speed_error / tol)
        speed_term = self.config["relative_speed_bonus"] * relative_speed_score

        # 2) Changement de voie
        lane_left = self.action_type.actions_indexes["LANE_LEFT"]
        lane_right = self.action_type.actions_indexes["LANE_RIGHT"]
        faster = self.action_type.actions_indexes["FASTER"]
        slower = self.action_type.actions_indexes["SLOWER"]

        lane_change_term = (
            -self.config["lane_change_penalty"]
            if action in [lane_left, lane_right]
            else 0.0
        )

        # 3) Petite pénalité si on reste à l'extrême gauche
        lane_id = self.vehicle.lane_index[2]
        left_lane_term = -self.config["left_lane_penalty"] if lane_id == 0 else 0.0

        # 4) Bonus de dépassement utile
        overtake_term = 0.0
        if front_vehicle is not None:
            current_front_x = front_vehicle.position[0] - self.vehicle.position[0]
            if self.prev_front_x is not None and current_front_x > self.prev_front_x + 1.0:
                overtake_term = self.config["overtake_bonus"]
            self.prev_front_x = current_front_x
        else:
            self.prev_front_x = None

        # 5) Confort longitudinal
        acceleration_term = -self.config["acceleration_penalty_weight"] * abs(acceleration)
        jerk_term = -self.config["jerk_penalty_weight"] * abs(jerk)

        # 6) Petite pénalité sur FASTER / SLOWER
        action_acceleration_term = (
            -self.config["action_acceleration_penalty"]
            if action in [faster, slower]
            else 0.0
        )

        # 7) Bonus si la vitesse reste lisse
        stable_speed_term = (
            self.config["stable_speed_bonus"]
            if abs(acceleration) < 0.4
            else 0.0
        )

        reward = (
            collision_term
            + speed_term
            + lane_change_term
            + left_lane_term
            + overtake_term
            + acceleration_term
            + jerk_term
            + action_acceleration_term
            + stable_speed_term
        )

        self.prev_speed = speed
        self.prev_acceleration = acceleration

        reward *= float(self.vehicle.on_road)

        return float(reward)