import time

import pygame
import sys

from game.View import View
from samegamerl.game.game import Game  # Import the game logic from another file
from samegamerl.game.game_config import GameFactory


# initialize Pygame
pygame.init()

# Use medium game configuration by default
config = GameFactory.large()

# starting the game_logic
game = Game(config)

view = View(game)

# create the screen based on config
screen = pygame.display.set_mode((view.screen_width, view.screen_height))
pygame.display.set_caption("SameGame")

# Call the game logic function from the other file
running = True
while running and not game.done():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            view.click(event)
            screen.fill((255, 255, 255))
            view.draw(screen, game.get_board())
            pygame.display.flip()
            print(game.get_singles())

    screen.fill((255, 255, 255))
    view.draw(screen, game.get_board())
    pygame.display.flip()


# Quit Pygame
pygame.quit()
sys.exit()
