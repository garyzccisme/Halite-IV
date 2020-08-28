import random

import numpy as np

from helper import *
from kaggle_helpers import *


class BronzeBot:

    def __init__(self, obs, config):
        self.obs = obs
        self.config = config
        self.board = Board(obs, config)
        self.size = config.size
        self.me = self.board.current_player

        self.SHIP_ACTION_DICT = {
            (1, 0): ShipAction.EAST,
            (-1, 0): ShipAction.WEST,
            (0, 1): ShipAction.NORTH,
            (0, -1): ShipAction.SOUTH
        }

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
    def radar(self, unit: Ship, dis: int = 2):
        """
        Radar Scanning for ship & shipyard.
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

                scan_pos = unify_pos(pos + (x, y), self.size)
                cell = self.board[scan_pos]
                halite[scan_pos] = cell.halite

                if scan_pos != unit.position and (cell.ship or cell.shipyard):
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
                    # To be determined
                    # Estimate halite gain in scan_pos with t [2, dis + 1] turns.
                    # The closer the cell, the more turns ship has to collect halite.
                    free_halite[scan_pos] = []
                    for t in range(2, dis + 2):
                        free_halite[scan_pos].append(
                            estimate_gain(cell.halite, cal_dis(pos, scan_pos), t)
                        )

        self.unit_radar[unit.id] = {
            'dis': dis,
            'halite': halite,
            'free_halite': free_halite,
            'ally_ship': ally_ship,
            'enemy_ship': enemy_ship,
            'ally_shipyard': ally_shipyard,
            'enemy_shipyard': enemy_shipyard,
        }

    def navigate(self, ship: Ship, des: Point):
        """
        Navigate ship to destination, give out optimal action for current turn.
        Args:
            ship: Ship.
            des: destination position.
        """
        candidate_action = []
        # There are actually 4 different paths, here we just choose the direct one
        move_x, move_y = unify_pos(des, self.size) - ship.position

        # Check direction availability & Add candidate action
        next_move = [(np.sign(move_x), 0), (0, np.sign(move_y))]
        for move in next_move:
            next_pos = ship.position + move
            next_cell = self.board[next_pos]

            condition_1 = next_cell.ship and next_cell.ship.player != self.me and next_cell.ship.halite > ship.halite
            condition_2 = next_cell.shipyard and next_cell.shipyard.player != self.me
            condition_3 = self.security_check(ship, dis=1, pos=next_pos)

            if (condition_1 or condition_2) and condition_3:
                candidate_action.append(self.SHIP_ACTION_DICT[move])

        # Randomly choose an action
        if candidate_action:
            ship.next_action = random.choice(candidate_action)

    def course_reversal(self, ship: Ship):
        """
        Command function for DEPOSIT ship navigation.
        """
        shipyard_pos = np.array(np.where(self.unit_map >= 2)).T
        # TODO: refactor with cal_dis(x, y)
        nearest_shipyard_x, nearest_shipyard_y = shipyard_pos[
            np.argmin(np.abs(shipyard_pos - ship.position).sum(axis=1))
        ]
        self.navigate(ship, Point(nearest_shipyard_x, nearest_shipyard_y))

    def security_check(self, ship: Ship, dis: int = 1, pos: Point = None) -> list:
        """
        Check if ship is clear in given distance.
        Args:
            ship: Ship
            dis: Int, Default = 1. The distance of security_check.
            pos: Point, Default = None. If is given, then take pos as the security check center.

        Returns: List of close enemy, if clear then return an empty list.
        """
        radar = self.unit_radar[ship.id]
        close_enemy = []
        if not pos:
            pos = ship.position
        for enemy_pos in radar['enemy_ship']:
            enemy_ship = self.board[enemy_pos]
            if cal_dis(pos, enemy_pos) <= dis and ship.halite >= enemy_ship.halite:
                close_enemy.append(enemy_pos)
        return close_enemy

    def explore_command(self, ship: Ship, radar: dict):
        """
        Command function for EXPLORE.

        Strategy 1: if ship state is EXPLORE, navigate to the position with max free halite.
        Strategy 2: if ship is in the max free halite position, turn EXPLORE to COLLECT.
        """
        max_free_halite = np.max(list(radar['free_halite'].values()))

        # Check if ship has arrived max free halite position
        if radar['free_halite'][ship.position][-1] == max_free_halite:
            # Change ship state, ship.next_action = None
            self.ship_state[ship.id] = 'COLLECT'
        else:
            # If there's no halite, expand radar distance
            if max_free_halite == 0:
                self.ship_command(ship, radar['dis'] + 1)
            else:
                candidate = []
                for pos, free_halite in radar['free_halite'].items():
                    if free_halite[-1] == max_free_halite:
                        candidate.append(pos)

                # Randomly choose a destination from candidate
                des = random.choice(candidate)
                self.navigate(ship, des)

    def ship_command(self, ship: Ship, radar_dis: int = 2, deposit_halite: int = 500, security_dis: int = 1):
        """
        For each turn, update action of each ship.

        Args:
            ship: Ship
            radar_dis: The radar scan distance of ship.
            deposit_halite: The threshold halite value for ship to hold.
            security_dis: The distance for security check.
        """
        # Before giving action, do radar first
        self.radar(ship, radar_dis)
        radar = self.unit_radar[ship.id]

        # Assign EXPLORE to new ship
        if ship.id not in self.ship_state:
            self.ship_state[ship.id] = 'EXPLORE'

        # DEPOSIT
        # Strategy 1: if ship is in a shipyard, and ship.halite is 0, turn DEPOSIT to EXPLORE
        # Strategy 2: if ship state is DEPOSIT, navigate to nearest shipyard
        # Strategy 3: if ship halite is lower than deposit_halite and radar is clear, turn DEPOSIT to EXPLORE
        if self.ship_state[ship.id] == 'DEPOSIT':

            # If ship has deposited halite to shipyard, assign EXPLORE to ship.
            if ship.cell.shipyard and ship.halite == 0:
                self.ship_state[ship.id] = 'EXPLORE'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)

            if ship.halite >= deposit_halite:
                self.course_reversal(ship)
            else:
                if not self.security_check(ship, security_dis):
                    self.ship_state[ship.id] = 'EXPLORE'
                    self.ship_command(ship, radar_dis, deposit_halite, security_dis)
                else:
                    # Enemy is close, stick to DEPOSIT ship state.
                    self.course_reversal(ship)

        # EXPLORE
        elif self.ship_state[ship.id] == 'EXPLORE':
            self.explore_command(ship, radar)

        # COLLECT
        # Strategy 1: if ship halite reaches deposit_halite, turn COLLECT to DEPOSIT
        # Strategy 2: if enemy ship shows in radar, turn COLLECT TO DEPOSIT
        elif self.ship_state[ship.id] == 'COLLECT':
            if ship.halite >= deposit_halite:
                self.ship_state[ship.id] = 'DEPOSIT'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                if not self.security_check(ship, security_dis):
                    self.explore_command(ship, radar)
                else:
                    self.ship_state[ship.id] = 'DEPOSIT'
                    self.course_reversal(ship)

    def spawn_command(self, max_ship: int = 5):
        """
        Command function for shipyard to SPAWN ship.

        Strategy: keep ship number in max_ship.
        Args:
            max_ship: The upper limit of ships.
        """
        empty_shipyard = [shipyard for shipyard in self.me.shipyards if not shipyard.cell.ship]
        new_ship = 0
        while len(self.me.ships) + new_ship < max_ship and len(empty_shipyard) > 0:
            shipyard = empty_shipyard.pop(0)
            shipyard.next_action = ShipyardAction.SPAWN
            new_ship += 1

    def convert_command(self):
        """
        Command function for ship to CONVERT to shipyard.

        Strategy: if there's no shipyard, randomly pick a ship with min cell halite to convert.
        """
        if not self.me.shipyards:
            min_cell_halite = np.min([ship.cell.halite for ship in self.me.ships])
            ship_candidate = [ship for ship in self.me.ships if ship.cell.halite == min_cell_halite]
            convert_ship = random.choice(ship_candidate)
            convert_ship.next_action = ShipAction.CONVERT
            self.ship_state[convert_ship.id] = 'CONVERT'

    def play(self, radar_dis=2, deposit_halite=500, security_dis=1, max_ship=5):
        """
        Main Function
        """
        print('MY TURN')
        self.update_map()

        print('- update map')

        self.convert_command()
        print('- convert command ')
        for ship in self.me.ships:
            print('-- command {}'.format(ship.id))
            self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            print('---- ship state: {}'.format(self.ship_state[ship.id]))
            print('---- ship action: {}'.format(ship.next_action))

        self.spawn_command(max_ship)
        print('- spawn command')

        return self.me.next_actions
