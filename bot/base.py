from kaggle_helpers import *


class Bot:
    def __init__(self, obs, config):
        self.obs = obs
        self.config = config
        self.board = Board(obs, config)
        self.size = config.size
        self.me = self.board.current_player

    def play(self):
        raise NotImplementedError
