import pygame
from flightgame.game.game_params import COLORS, NUM_COLS, TILE_SIZE


class Tile:

    def __init__(self, x, y, size, color, game, row, col):
        self.rect = pygame.Rect(x, y, size, size)
        self.color = color
        self.game = game
        self.row = row
        self.col = col

    def click(self, event):
        if self.rect.collidepoint(event.pos):
            if self.color == (255, 255, 255):
                return
            self.game.move((self.row, self.col))
            return True

    def draw(self, screen):
        pygame.draw.rect(screen, COLORS[self.color], self.rect)

    def draw_with_value(self, screen, values):
        pygame.draw.rect(screen, COLORS[self.color], self.rect)
        font = pygame.font.Font(None, 24)
        text = font.render(self.first_n_digits(values[0][self.row*NUM_COLS + self.col], 5), True, (170, 170, 170))
        screen.blit(text, (self.rect.x + 5, self.rect.y + TILE_SIZE/2-5))

    def first_n_digits(self, num, n):
        num_str = str(num)

        if n >= len(num_str):
            return num_str

        if num < 0:
            return num_str[:(n+1)]

        return num_str[:n]
