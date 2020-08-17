import random

from kaggle_environments.envs.halite.helpers import *
import numpy as np

from helpers import *


class BronzeBot:

    def __init__(self, obs, config):
        self.obs = obs
        self.config = config
        self.board = Board(obs, config)
        self.size = config.size
        self.me = self.board.current_player

        self.halite_map = None
        self.unit_map = None

        self.unit_radar = {}
        self.radar_params = {}
        self.ship_state = {}

    def update_map(self):
        """
        In the beginning of each turn, update halite & unit map.
        """
        # Rotate the halite map so that 2-D Point can be used as an array index
        self.halite_map = np.rot90(np.reshape(self.board.observation['halite'], (self.size, self.size)), k=3)

        # Initialize unit map with all zeros
        self.unit_map = np.zeros((self.size, self.size))
        for i, (_, shipyards, ships) in enumerate(self.board.observation['players']):
            if i == self.me.id:
                for index in shipyards.values():
                    self.unit_map[index_to_position(index, self.size)] += 2
                for index, _ in ships.values():
                    self.unit_map[index_to_position(index, self.size)] += 1
            else:
                for index in shipyards.values():
                    self.unit_map[index_to_position(index, self.size)] += -2
                for index, _ in ships.values():
                    self.unit_map[index_to_position(index, self.size)] += -1

    # TODO: refactor for efficiency
    def radar(self, unit, dis=2):
        """
        Radar Scaning for ship & shipyard.
        Gather information of [ally, enemy, halite, free halite].
        Note: free halite here is estimated gain given number of turns in free area.
        Args:
            unit: ship or shipyard
            dis: Manhattan Distance for radar scanning
        """
        pos = unit.position
        halite, free_halite = {}, {}
        ally_ship, ally_shipyard = [], []
        enemy_ship, enemy_shipyard = [], []

        # Start scanning
        for x in range(-dis, dis + 1):
            for y in range(abs(x) - dis, dis - abs(x) + 1):

                scan_pos = pos + (x, y)
                cell = self.board[scan_pos]
                halite[scan_pos] = cell.halite

                if cell.ship or cell.shipyard:
                    if cell.ship:
                        if cell.ship.player == self.me:
                            ally_ship.append(cell.position)
                        else:
                            enemy_ship.append(cell.position)
                    if cell.shipyard:
                        if cell.shipyard.player == self.me:
                            ally_shipyard.append(cell.position)
                        else:
                            enemy_shipyard.append(cell.position)
                else:
                    # Estimate halite gain in scan_pos with t [2, dis + 1] turns.
                    # The closer the cell, the more turns ship has to collect halite.
                    free_halite[scan_pos] = []
                    for t in range(2, dis + 2):
                        free_halite[scan_pos].append(
                            estimate_gain(cell.halite, cal_dis(pos, scan_pos), t)
                        )

        self.unit_radar[unit.id] = {
            'halite': halite,
            'free_halite': free_halite,
            'ally_ship': ally_ship,
            'enemy_ship': enemy_ship,
            'ally_shipyard': ally_shipyard,
            'enemy_shipyard': enemy_shipyard,
        }

    def navigate(self, ship, dest):
        """
        Navigate ship to destination, give out optimal action for current turn.
        Args:
            ship: Ship
            dest: Point, destination position
        """
        return

    def deposit_command(self, ship):
        """
        Command function for DEPOSIT ship state.
        """
        shipyard_pos = np.array(np.where(self.unit_map >= 2)).T
        nearest_shipyard_pos = shipyard_pos[np.argmin(np.abs(shipyard_pos - ship.position).sum(axis=1))]
        self.navigate(ship, Point(tuple(nearest_shipyard_pos)))

    def security_check(self, ship, dis=1):
        """
        Check if ship is clear in given distance.
        """
        radar = self.unit_radar[ship.id]
        for enemy_pos in radar['enemy_ship']:
            if cal_dis(ship.position, enemy_pos) <= dis:
                return False
        return True

    def command(self, ship, radar_dis=2, deposit_halite=500):
        """
        For each turn, update action of each ship.
        """
        # Before giving action, do radar first
        self.radar(ship.position, radar_dis)
        radar = self.unit_radar[ship.id]

        # Strategy: if ship state is None (ship in shipyard), assign EXPLORE to ship state
        if ship.id not in self.ship_state:
            self.ship_state[ship.id] = 'EXPLORE'

        # Strategy 1: if ship state is DEPOSIT, navigate to nearest shipyard
        # Strategy 2: if ship halite is lower than deposit_halite and radar is clear, turn DEPOSIT to EXPLORE
        if self.ship_state[ship.id] == 'DEPOSIT':

            if ship.halite >= deposit_halite:
                self.deposit_command(ship)
            else:
                for enemy_pos in radar['enemy_ship']:
                    if cal_dis(ship.position, enemy_pos) == 1:
                        # Enemy is close, stick to DEPOSIT ship state
                        self.deposit_command(ship)
                        return
                self.ship_state[ship.id] = 'EXPLORE'
                self.command(ship, radar_dis, deposit_halite)

        # Strategy: if ship state is EXPLORE, navigate to the position with max free halite
        elif self.ship_state[ship.id] == 'EXPLORE':

            max_free_halite = np.max(list(radar['free_halite'].values()))

            # If there's no halite, expand radar distance
            if max_free_halite == 0:
                self.command(ship, radar_dis + 1)
            else:
                candidate = []
                for pos, free_halite in radar['free_halite']:
                    if free_halite[-1] == max_free_halite:
                        candidate.append(pos)

                # Randomly choose a destination from candidate
                des = random.choice(candidate)
                self.navigate(ship, Point(des))

        # Strategy 1: if ship halite reaches 500, turn COLLECT to DEPOSIT
        # Strategy 2: if enemy ship shows in radar, turn COLLECT TO DEPOSIT
        elif self.ship_state[ship.id] == 'COLLECT':
            return