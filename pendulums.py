import numpy as np
from constants import *


class PendulumCart:
    """Shared cart state/behavior used by both pendulum modes.

    The cart is driven only by the applied force and its own friction - the
    pendulum swinging on top of it does not push back on it. This keeps the
    cart's motion simple and fully predictable from the control input alone.
    """

    def __init__(self):
        self.reset()

    def reset(self, angle=0.0):
        self.x = 0.0
        self.x_dot = 0.0

    def cart_acceleration(self, force):
        limit = TRACK_HALF_WIDTH / PIXELS_PER_METER
        if self.x <= -limit and force < 0:
            self.x = -limit
            self.x_dot = 0.0
            return 0.0

        if self.x >= limit and force > 0:
            self.x = limit
            self.x_dot = 0.0
            return 0.0

        return (force - DAMPING_CART * self.x_dot) / CART_MASS

    def clamp_to_track(self):
        limit = TRACK_HALF_WIDTH / PIXELS_PER_METER
        if self.x < -limit:
            self.x, self.x_dot = -limit, 0.0
        elif self.x > limit:
            self.x, self.x_dot = limit, 0.0

    def cart_pixel_pos(self):
        return (SCREEN_W / 2 + self.x * PIXELS_PER_METER, TRACK_Y)


class SinglePendulum(PendulumCart):
    """Cart with a single inverted pendulum (point mass on a rod)."""

    name = "Single Pendulum"

    def nudge_tip(self, impulse):
        self.theta_dot += impulse

    def reset(self, angle=np.pi):
        super().reset()
        self.theta = angle
        self.theta_dot = 0.0

    def step(self, force, dt):
        g, l, m = GRAVITY, S_POLE_LEN, S_POLE_MASS
        s, c = np.sin(self.theta), np.cos(self.theta)

        x_ddot = self.cart_acceleration(force)

        theta_ddot = (g * s - c * x_ddot) / l - DAMPING_JOINT * \
            self.theta_dot / (m * l * l)

        self.x_dot += x_ddot * dt
        self.x += self.x_dot * dt
        self.theta_dot += theta_ddot * dt
        self.theta += self.theta_dot * dt
        self.x_dot = np.clip(self.x_dot, -MAX_X_DOT, MAX_X_DOT)
        self.theta_dot = np.clip(self.theta_dot, -MAX_THETA_DOT, MAX_THETA_DOT)
        self.clamp_to_track()

    def upright_fraction(self):
        return (np.cos(self.theta) + 1) / 2

    def joint_positions(self):
        cart_px = self.cart_pixel_pos()
        bob_px = (
            cart_px[0] + S_POLE_LEN * PIXELS_PER_METER * np.sin(self.theta),
            cart_px[1] - S_POLE_LEN * PIXELS_PER_METER * np.cos(self.theta),
        )
        return [cart_px, bob_px]

    def get_state(self):
        return [
            self.x / TRACK_LIMIT,
            self.x_dot / X_DOT_SCALE,
            np.sin(self.theta),
            np.cos(self.theta),
            self.theta_dot / THETA_DOT_SCALE,
        ]

    def get_fitness(self, points_per_tick):
        angle_error = abs(
            np.arctan2(
                np.sin(self.theta),
                np.cos(self.theta)))

        is_balanced = angle_error <= np.deg2rad(MAX_ANGLE_TO_AWARD_POINTS)
        upright = self.upright_fraction()

        position_error = abs(self.x) / TRACK_LIMIT
        angle_reward = np.exp(-4.0 * angle_error**2)
        angular_velocity_penalty = (abs(self.theta_dot) / THETA_DOT_SCALE)
        velocity_penalty = (abs(self.x_dot) / X_DOT_SCALE) ** 2

        fitness = (
            1.0 * angle_reward
            - 0.1 * position_error
            - 0.1 * velocity_penalty
            - 0.1 * angular_velocity_penalty
        )
        score = (upright if is_balanced else 0)

        return fitness, score*points_per_tick


class DoublePendulum(PendulumCart):
    """Cart with two linked inverted pendulum rods (point mass on each end)."""

    name = "Double Pendulum"

    def nudge_tip(self, impulse):
        self.theta2_dot += impulse

    def reset(self, angle=np.pi, angle2=np.pi):
        super().reset()
        self.theta1 = angle
        self.theta1_dot = 0.0
        self.theta2 = angle2
        self.theta2_dot = 0.0

    def step(self, force, dt):
        m1, m2 = D_POLE_MASS_1, D_POLE_MASS_2
        l1, l2 = D_POLE_LEN_1, D_POLE_LEN_2
        g = GRAVITY

        t1d, t2d = self.theta1_dot, self.theta2_dot
        s1, c1 = np.sin(self.theta1), np.cos(self.theta1)
        s2, c2 = np.sin(self.theta2), np.cos(self.theta2)
        s12 = np.sin(self.theta1 - self.theta2)
        c12 = np.cos(self.theta1 - self.theta2)

        x_ddot = self.cart_acceleration(force)

        # A = np.array([
        #    [(m1 + m2) * l1 * l1, m2 * l1 * l2 * c12],
        #    [m2 * l1 * l2 * c12,  m2 * l2 * l2],
        # ])
        # b = np.array([
        #    -m2 * l1 * l2 * s12 * t2d ** 2 + (m1 + m2) * g * l1 * s1
        #    - DAMPING_JOINT * t1d - (m1 + m2) * l1 * c1 * x_ddot,
        #    m2 * l1 * l2 * s12 * t1d ** 2 + m2 * g * l2 * s2
        #    - DAMPING_JOINT * t2d - m2 * l2 * c2 * x_ddot,
        # ])
        # t1_ddot, t2_ddot = np.linalg.solve(A, b)

        a11 = (m1 + m2) * l1 * l1
        a12 = m2 * l1 * l2 * c12
        a22 = m2 * l2 * l2

        b0 = (-m2 * l1 * l2 * s12 * t2d ** 2 + (m1 + m2) * g * l1 * s1
              - DAMPING_JOINT * t1d - (m1 + m2) * l1 * c1 * x_ddot)
        b1 = (m2 * l1 * l2 * s12 * t1d ** 2 + m2 * g * l2 * s2
              - DAMPING_JOINT * t2d - m2 * l2 * c2 * x_ddot)

        det = a11 * a22 - a12 * a12  # matrix is symmetric, a21 == a12
        t1_ddot = (b0 * a22 - a12 * b1) / det
        t2_ddot = (a11 * b1 - a12 * b0) / det

        self.x_dot += x_ddot * dt
        self.x += self.x_dot * dt
        self.theta1_dot += t1_ddot * dt
        self.theta1 += self.theta1_dot * dt
        self.theta2_dot += t2_ddot * dt
        self.theta2 += self.theta2_dot * dt
        self.x_dot = np.clip(self.x_dot, -MAX_X_DOT, MAX_X_DOT)
        self.theta1_dot = np.clip(
            self.theta1_dot, -MAX_THETA_DOT, MAX_THETA_DOT)
        self.theta2_dot = np.clip(
            self.theta2_dot, -MAX_THETA_DOT, MAX_THETA_DOT)
        self.clamp_to_track()

    def upright_fraction(self):
        """1.0 when both links point straight up, 0.0 when both hang down.

        Based on the height of the tip (end of the second link) relative to
        the cart, normalized by the pendulum's total length.
        """
        tip_height = D_POLE_LEN_1 * \
            np.cos(self.theta1) + D_POLE_LEN_2 * np.cos(self.theta2)
        total_length = D_POLE_LEN_1 + D_POLE_LEN_2
        return (tip_height / total_length + 1) / 2

    def joint_positions(self):
        """Pixel positions from the cart out to the tip: [cart, joint1, tip]."""
        cart_px = self.cart_pixel_pos()
        j1_px = (
            cart_px[0] + D_POLE_LEN_1 * PIXELS_PER_METER * np.sin(self.theta1),
            cart_px[1] - D_POLE_LEN_1 * PIXELS_PER_METER * np.cos(self.theta1),
        )
        tip_px = (
            j1_px[0] + D_POLE_LEN_2 * PIXELS_PER_METER * np.sin(self.theta2),
            j1_px[1] - D_POLE_LEN_2 * PIXELS_PER_METER * np.cos(self.theta2),
        )
        return [cart_px, j1_px, tip_px]

    def get_state(self):
        return [
            self.x / TRACK_LIMIT,
            self.x_dot / X_DOT_SCALE,
            np.sin(self.theta1),
            np.cos(self.theta1),
            self.theta1_dot / THETA_DOT_SCALE,

            np.sin(self.theta2),
            np.cos(self.theta2),
            self.theta2_dot / THETA_DOT_SCALE
        ]

    def get_fitness(self, points_per_tick):
        upright = self.upright_fraction()  # 0 to 1, continuous

        angle_error1 = abs(np.arctan2(
            np.sin(self.theta1), np.cos(self.theta1)))
        angle_error2 = abs(np.arctan2(
            np.sin(self.theta2), np.cos(self.theta2)))
        is_balanced = (angle_error1 <= np.deg2rad(MAX_ANGLE_TO_AWARD_POINTS)
                       and angle_error2 <= np.deg2rad(MAX_ANGLE_TO_AWARD_POINTS))

        # joint_misalignment = abs(
        #    np.arctan2(np.sin(self.theta1 - self.theta2),
        #               np.cos(self.theta1 - self.theta2))
        # ) / np.pi

        position_penalty = (abs(self.x) / TRACK_LIMIT) ** 2
        velocity_penalty = (abs(self.x_dot) / X_DOT_SCALE) ** 2

        # Normalized combined angular speed, 0 (still) to ~1+ (fast)
        angular_speed = (abs(self.theta1_dot) +
                         abs(self.theta2_dot)) / (2 * THETA_DOT_SCALE)

        # KEY CHANGE: gate the upright reward by stillness, don't just subtract a
        # separate velocity penalty. A fast-spinning pendulum passing through
        # upright now gets heavily discounted credit instead of full credit.
        stillness_factor = np.exp(-3.0 * angular_speed ** 2)
        upright_reward = upright * stillness_factor

        fitness = (
            1.0 * upright_reward
            # - 0.3 * joint_misalignment
            - 0.05 * position_penalty
            - 0.05 * velocity_penalty
        )

        score = (upright if is_balanced else 0)

        return fitness, score * points_per_tick
