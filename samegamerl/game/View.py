import pygame
from samegamerl.game.Tile import Tile
from samegamerl.game.game_params import (
    NUM_ROWS,
    NUM_COLS,
    NUM_COLORS,
    COLORS,
    TILE_SIZE,
    GAP,
    CONTROL_GAP,
    FIELD_CONTROL_MARGIN,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)


class View:

    def __init__(self, game_logic):
        self.game = game_logic
        self.tiles = []

    def draw_board(self, screen, board):
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                color = board[row][col]
                self.tiles.append(
                    Tile(
                        col * TILE_SIZE + GAP,
                        row * TILE_SIZE + GAP,
                        TILE_SIZE,
                        color,
                        self.game,
                        row,
                        col,
                    )
                )
                # pygame.draw.rect(screen, color, (col * TILE_SIZE + GAP, row * TILE_SIZE + GAP, TILE_SIZE, TILE_SIZE))

    def click(self, event):
        found = False
        for tile in self.tiles:
            if not found:
                found = tile.click(event)

    def draw(self, screen, board):
        self.draw_board(screen, board)
        for tile in self.tiles:
            tile.draw(screen)

    def draw_text(self, screen, board, values):
        self.draw_board(screen, board)
        for tile in self.tiles:
            tile.draw_with_value(screen, values)

    def show_score(self, screen, score):
        font = pygame.font.Font(None, 100)  # Font and font size
        text = font.render(
            "Game over!", True, (255, 255, 255)
        )  # Text, antialiasing, and text color
        text_rect = text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50 - 50)
        )
        result_text = font.render(
            str(score[0]) + " - " + str(score[1]), True, (255, 255, 255)
        )
        result_text_rect = result_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50 - 50)
        )
        # text_rect = tex.Rect(screen, (255, 255, 255), (2*TILE_SIZE, 2*TILE_SIZE, 3*TILE_SIZE, 2*TILE_SIZE))
        screen.blit(text, text_rect)
        screen.blit(result_text, result_text_rect)
