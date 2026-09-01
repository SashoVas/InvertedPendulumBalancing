SCREEN_W, SCREEN_H = 900, 600
TRACK_Y = SCREEN_H // 2  # vertical screen position of the track
TRACK_HALF_WIDTH = 380  # how far (in pixels) the cart may travel
PIXELS_PER_METER = 120

FPS = 60
MAX_DT = 0.05  # clamp large frame-time spikes
SUBSTEPS = 4  # physics substeps per frame (stability)

GRAVITY = 9.81
CART_MASS = 1.0
FORCE_MAG = 20.0  # push force applied by arrow keys
DAMPING_CART = 0.5  # friction on the cart
DAMPING_JOINT = 0.02  # friction at each pendulum joint

ROUND_DURATION = 15.0  # length of a round, in seconds
SCORE_TICK = 0.1  # seconds between score updates
POINTS_PER_TICK = 0.1  # points per tick when perfectly upright

# Single pendulum parameters
S_POLE_MASS = 1.0
S_POLE_LEN = 1.6

# Double pendulum parameters (two links, each with a point mass at its end)
# D_POLE_MASS_1 = 0.6
# D_POLE_MASS_2 = 0.6
D_POLE_MASS_1 = 0.6  # 0.3  # 0.6  # 1.0  # 1.5  # was 0.6
D_POLE_MASS_2 = 0.6  # 0.3  # 0.3  # 0.3  # 0.3  # was 0.6

D_POLE_LEN_1 = 1.0
D_POLE_LEN_2 = 1.0

# Colors
BG_COLOR = (18, 18, 24)
TRACK_COLOR = (70, 70, 80)
CART_COLOR = (230, 230, 240)
LINK_COLORS = [(240, 90, 90), (90, 160, 240)]
TEXT_COLOR = (230, 230, 230)
HIGHLIGHT_COLOR = (255, 210, 90)


TRACK_LIMIT = TRACK_HALF_WIDTH / PIXELS_PER_METER
X_DOT_SCALE = 5.0
THETA_DOT_SCALE = 20.0

MAX_ANGLE_TO_AWARD_POINTS = 45

SPACE_NUDGE_IMPULSE = 0.75

# Standard deviation (radians) of the angle noise applied to each link when a
# round starts upright instead of hanging. ~0.10 rad is about 6 degrees.
UPRIGHT_START_NOISE = 0.10

# A round that starts upright ends as soon as the pendulum has fallen this far,
# rather than simulating the rest of the round. upright_fraction() is 1.0 when
# vertical, 0.5 when the tip is level with the cart, 0.0 when hanging.
# Only applies when start_upright=True - from a hanging start it would fire at t=0.
FALL_TERMINATION_UPRIGHT = 0.4
