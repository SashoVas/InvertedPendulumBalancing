import math

import numpy as np
from constants import *


# Derived double-pendulum constants, precomputed at import.
#
# The mass matrix of the two-link system is
#
#     A = [[(m1 + m2) * l1**2,           m2 * l1 * l2 * cos(t1 - t2)],
#          [m2 * l1 * l2 * cos(t1 - t2), m2 * l2**2                 ]]
#
# Both diagonal entries are constant - only the off-diagonal varies with the
# state - so the 2x2 system is solved in closed form (Cramer's rule) below
# rather than with np.linalg.solve, whose dispatch overhead dwarfs the
# arithmetic at this size.
_A11 = (D_POLE_MASS_1 + D_POLE_MASS_2) * D_POLE_LEN_1 * D_POLE_LEN_1
_A22 = D_POLE_MASS_2 * D_POLE_LEN_2 * D_POLE_LEN_2
_A11_A22 = _A11 * _A22
_M2_L1_L2 = D_POLE_MASS_2 * D_POLE_LEN_1 * D_POLE_LEN_2
_M12_G_L1 = (D_POLE_MASS_1 + D_POLE_MASS_2) * GRAVITY * D_POLE_LEN_1
_M2_G_L2 = D_POLE_MASS_2 * GRAVITY * D_POLE_LEN_2
_M12_L1 = (D_POLE_MASS_1 + D_POLE_MASS_2) * D_POLE_LEN_1
_M2_L2 = D_POLE_MASS_2 * D_POLE_LEN_2

_MAX_ANGLE_RAD = math.radians(MAX_ANGLE_TO_AWARD_POINTS)


class PendulumCart:
    """Shared cart state/behavior used by both pendulum modes.

    The cart is driven only by the applied force and its own friction - the
    pendulum swinging on top of it does not push back on it. This keeps the
    cart's motion simple and fully predictable from the control input alone.
    """

    def __init__(self):
        self.reset()

    def reset(self):
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

    def reset(self, upright=False, rng=None):
        """Start hanging (theta = pi), or upright with a little angle noise."""
        super().reset()
        if upright:
            if rng is None:
                rng = np.random.default_rng()
            self.theta = float(rng.normal(0.0, UPRIGHT_START_NOISE))
        else:
            self.theta = np.pi
        self.theta_dot = 0.0

    def step(self, force, dt):
        g, l, m = GRAVITY, S_POLE_LEN, S_POLE_MASS
        s, c = np.sin(self.theta), np.cos(self.theta)

        x_ddot = self.cart_acceleration(force)

        theta_ddot = (g * s - c * x_ddot) / l - DAMPING_JOINT * self.theta_dot / (
            m * l * l
        )

        self.x_dot += x_ddot * dt
        self.x += self.x_dot * dt
        self.theta_dot += theta_ddot * dt
        self.theta += self.theta_dot * dt

        self.clamp_to_track()

    def upright_fraction(self):
        return (math.cos(self.theta) + 1) / 2

    def has_fallen(self):
        return self.upright_fraction() < FALL_TERMINATION_UPRIGHT

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
        angle_error = abs(math.remainder(self.theta, math.tau))

        is_balanced = angle_error <= _MAX_ANGLE_RAD
        upright = self.upright_fraction()

        position_error = abs(self.x) / TRACK_LIMIT
        angle_reward = math.exp(-4.0 * angle_error * angle_error)
        angular_velocity_penalty = abs(self.theta_dot) / THETA_DOT_SCALE
        velocity_penalty = (abs(self.x_dot) / X_DOT_SCALE) ** 2

        fitness = (
            1.0 * angle_reward
            - 0.1 * position_error
            - 0.1 * velocity_penalty
            - 0.1 * angular_velocity_penalty
        )
        score = upright if is_balanced else 0

        return fitness, score * points_per_tick


class DoublePendulum(PendulumCart):
    """Cart with two linked inverted pendulum rods (point mass on each end)."""

    name = "Double Pendulum"

    def nudge_tip(self, impulse):
        self.theta2_dot += impulse

    def reset(self, upright=False, rng=None):
        """Start hanging (both links at pi), or upright with a little angle noise."""
        super().reset()
        if upright:
            if rng is None:
                rng = np.random.default_rng()
            self.theta1 = float(rng.normal(0.0, UPRIGHT_START_NOISE))
            self.theta2 = float(rng.normal(0.0, UPRIGHT_START_NOISE))
        else:
            self.theta1 = np.pi
            self.theta2 = np.pi
        self.theta1_dot = 0.0
        self.theta2_dot = 0.0

    def angular_acceleration(self, t1, t2, t1d, t2d, x_ddot):
        """Angular accelerations of both links for an arbitrary state.

        The state is passed in rather than read off self so that the RK4
        integrator in step() can evaluate it at trial points along the step.
        """
        s1, c1 = math.sin(t1), math.cos(t1)
        s2, c2 = math.sin(t2), math.cos(t2)

        s12 = s1 * c2 - c1 * s2
        c12 = c1 * c2 + s1 * s2

        a12 = _M2_L1_L2 * c12

        b1 = (
            -_M2_L1_L2 * s12 * t2d * t2d
            + _M12_G_L1 * s1
            - DAMPING_JOINT * t1d
            - _M12_L1 * c1 * x_ddot
        )
        b2 = (
            _M2_L1_L2 * s12 * t1d * t1d
            + _M2_G_L2 * s2
            - DAMPING_JOINT * t2d
            - _M2_L2 * c2 * x_ddot
        )

        inv_det = 1.0 / (_A11_A22 - a12 * a12)

        return (
            (b1 * _A22 - a12 * b2) * inv_det,
            (_A11 * b2 - a12 * b1) * inv_det,
        )

    def step(self, force, dt):
        x_ddot = self.cart_acceleration(force)

        t1, t2 = self.theta1, self.theta2
        v1, v2 = self.theta1_dot, self.theta2_dot
        half = dt * 0.5
        a1, a2 = self.angular_acceleration(t1, t2, v1, v2, x_ddot)

        v1_b, v2_b = v1 + half * a1, v2 + half * a2
        b1, b2 = self.angular_acceleration(
            t1 + half * v1, t2 + half * v2, v1_b, v2_b, x_ddot
        )

        v1_c, v2_c = v1 + half * b1, v2 + half * b2
        c1, c2 = self.angular_acceleration(
            t1 + half * v1_b, t2 + half * v2_b, v1_c, v2_c, x_ddot
        )

        v1_d, v2_d = v1 + dt * c1, v2 + dt * c2
        d1, d2 = self.angular_acceleration(
            t1 + dt * v1_c, t2 + dt * v2_c, v1_d, v2_d, x_ddot
        )

        sixth = dt / 6.0
        self.theta1 = t1 + sixth * (v1 + 2.0 * v1_b + 2.0 * v1_c + v1_d)
        self.theta2 = t2 + sixth * (v2 + 2.0 * v2_b + 2.0 * v2_c + v2_d)
        self.theta1_dot = v1 + sixth * (a1 + 2.0 * b1 + 2.0 * c1 + d1)
        self.theta2_dot = v2 + sixth * (a2 + 2.0 * b2 + 2.0 * c2 + d2)

        self.x_dot += x_ddot * dt
        self.x += self.x_dot * dt

        self.clamp_to_track()

    def upright_fraction(self):
        tip_height = D_POLE_LEN_1 * math.cos(self.theta1) + D_POLE_LEN_2 * math.cos(
            self.theta2
        )
        total_length = D_POLE_LEN_1 + D_POLE_LEN_2
        return (tip_height / total_length + 1) / 2

    def link_fractions(self):
        """upright_fraction of each link on its own: 1.0 up, 0.5 horizontal, 0.0 down."""
        return (
            (math.cos(self.theta1) + 1) / 2,
            (math.cos(self.theta2) + 1) / 2,
        )

    def has_fallen(self):
        first, second = self.link_fractions()
        return first < FALL_TERMINATION_UPRIGHT or second < FALL_TERMINATION_UPRIGHT

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
            self.theta2_dot / THETA_DOT_SCALE,
        ]

    def get_fitness(self, points_per_tick):
        a1 = abs(math.remainder(self.theta1, math.tau))
        a2 = abs(math.remainder(self.theta2, math.tau))

        upright = self.upright_fraction()

        balanced = math.exp(-4.0 * a1 * a1) * math.exp(-4.0 * a2 * a2)
        fitness = (
            0.4 * upright
            + 0.6 * balanced
            - 0.05 * abs(self.x) / TRACK_LIMIT
            - 0.05 * (abs(self.theta1_dot) +
                      abs(self.theta2_dot)) / THETA_DOT_SCALE
        )

        is_balanced = max(a1, a2) <= _MAX_ANGLE_RAD
        return (
            max(fitness, 0.0),
            (upright if is_balanced else 0.0) * points_per_tick,
        )
