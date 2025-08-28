"""Game configuration system for flexible SameGameRL experimentation."""

from dataclasses import dataclass


@dataclass
class GameConfig:
    """num_colors defines the total number of colors on the board including the white color for the empty cells. Therefore the playable colors will be one less than num_colors"""

    num_rows: int = 8
    num_cols: int = 8
    num_colors: int = 4

    @property
    def total_cells(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def action_space_size(self) -> int:
        return self.total_cells

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.num_colors, self.num_rows, self.num_cols)

    def validate(self):
        if self.num_rows <= 0 or self.num_cols <= 0:
            raise ValueError("Board dimensions must be positive")
        if self.num_colors < 2:
            raise ValueError("Must have at least 2 colors")
        if self.num_colors > 6:
            raise ValueError("Maximum 6 colors supported for display")


class GameFactory:

    @staticmethod
    def small() -> GameConfig:
        config = GameConfig(num_rows=5, num_cols=5, num_colors=3)
        config.validate()
        return config

    @staticmethod
    def medium() -> GameConfig:
        config = GameConfig(num_rows=8, num_cols=8, num_colors=4)
        config.validate()
        return config

    @staticmethod
    def large() -> GameConfig:
        config = GameConfig(num_rows=15, num_cols=15, num_colors=6)
        config.validate()
        return config

    @staticmethod
    def default() -> GameConfig:
        return GameFactory.medium()

    @staticmethod
    def custom(num_rows: int, num_cols: int, num_colors: int) -> GameConfig:
        config = GameConfig(num_rows=num_rows, num_cols=num_cols, num_colors=num_colors)
        config.validate()
        return config
