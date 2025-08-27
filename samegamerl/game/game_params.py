# View constants
# Note: Game dimensions (NUM_ROWS, NUM_COLS, NUM_COLORS) have been moved to GameConfig
# Use GameFactory.small(), GameFactory.medium(), GameFactory.large() for standard configurations
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
# Screen dimensions are now calculated dynamically in View class based on GameConfig
NEIGHBOURS = []
