import time

import pygame
import sys

from game.View import View
from flightgame.game.game import Game  # Import the game logic from another file
from game.game_params import SCREEN_WIDTH, SCREEN_HEIGHT, NUM_ROWS, NUM_COLS, NUM_COLORS



# initialize Pygame
pygame.init()


# create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flight Game")



# starting the game_logic
game = gameLogic() # screen)

view = View(game)

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
            data = game.trainable_game()
            print("transformed data")
            prediction = model(tf.expand_dims(data, 0), training=False)
            print("made prediction")
            print(prediction)
            move = np.argmax(prediction)
            game.move(agent.play_test(game))
    """
    game.move(bot1.play(game))
    screen.fill((255, 255, 255))
    view.draw(screen, game.get_board())
    pygame.display.flip()
    game.move(bot2.play(game))
    
    screen.fill((255, 255, 255))
    view.draw(screen, game.get_board())
    pygame.display.flip()

    print("rendered")
    pygame.time.wait(5000)
    print("waited")
    game.move([10, 1])
    pygame.display.flip()
    """

    screen.fill((255, 255, 255))
    view.draw(screen, game.get_board())
    pygame.display.flip()


# Quit Pygame
pygame.quit()
sys.exit()