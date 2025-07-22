NUM_ROWS = 8  # Number of rows in the upper section
NUM_COLS = 8  # Number of columns in the upper section
NUM_COLORS = 4  # Number of extra squares in the lower section

# View constants
COLORS = [
    (255, 255, 255),  # White
    (252, 15, 15),  # Red
    (17, 181, 11),  # Green
    (11, 51, 181),  # Blue
    (252, 197, 15),  # Yellow
    (122, 11, 181),  # Magenta
    (0, 0, 0),  # Black
]

TILE_SIZE = 50  # Size of each square tile
GAP = 30  # Gap between the two sections
FIELD_CONTROL_MARGIN = 30
CONTROL_GAP = 20
SCREEN_WIDTH, SCREEN_HEIGHT = (
    GAP + TILE_SIZE * NUM_COLS + GAP,
    GAP + TILE_SIZE * NUM_ROWS + FIELD_CONTROL_MARGIN + TILE_SIZE + GAP,
)
NEIGHBOURS = []
