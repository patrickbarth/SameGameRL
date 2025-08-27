import pygame
from samegamerl.game.Tile import Tile
from samegamerl.game.game_params import (
    COLORS,
    TILE_SIZE,
    GAP,
    CONTROL_GAP,
    FIELD_CONTROL_MARGIN,
)
from samegamerl.game.game_config import GameConfig


class View:

    def __init__(self, game_logic):
        self.game = game_logic
        self.config = game_logic.config
        self.tiles = []
        
        # Calculate screen dimensions based on config
        self.screen_width = GAP + TILE_SIZE * self.config.num_cols + GAP
        self.screen_height = (
            GAP + TILE_SIZE * self.config.num_rows + FIELD_CONTROL_MARGIN + TILE_SIZE + GAP
        )

    def draw_board(self, screen, board):
        for row in range(self.config.num_rows):
            for col in range(self.config.num_cols):
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
            center=(self.screen_width // 2, self.screen_height // 2 - 50 - 50)
        )
        result_text = font.render(
            str(score[0]) + " - " + str(score[1]), True, (255, 255, 255)
        )
        result_text_rect = result_text.get_rect(
            center=(self.screen_width // 2, self.screen_height // 2 + 50 - 50)
        )
        # text_rect = tex.Rect(screen, (255, 255, 255), (2*TILE_SIZE, 2*TILE_SIZE, 3*TILE_SIZE, 2*TILE_SIZE))
        screen.blit(text, text_rect)
        screen.blit(result_text, result_text_rect)
