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
        self.ship_next_pos = set()

    # TODO: legacy function
    def get_map(self):
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

    # TODO: Figure out self collision and endless waiting
    def navigate(self, ship: Ship, des: Point):
        """
        Navigate ship to destination, give out optimal action for current turn.
        Args:
            ship: Ship.
            des: destination position.
        """
        # There are actually 4 different paths, here we just choose the direct one
        move_x, move_y = unify_pos(des, self.size) - ship.position

        candidate_move = []
        dangerous_move = []
        for move in [(np.sign(move_x), 0), (0, np.sign(move_y))]:
            if move != (0, 0):
                pos_access = self.case_analysis(ship, move)
                if pos_access == 'MOVE':
                    candidate_move.append(move)
                elif pos_access == 'DETOUR':
                    dangerous_move.append(move)

        # Randomly choose an action in candidate_move
        # If both candidate_move & dangerous_move are None, then the ship's order is WAIT
        if candidate_move:
            ship.next_action = self.SHIP_ACTION_DICT[random.choice(candidate_move)]
        elif dangerous_move:
            for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if move not in dangerous_move:
                    if self.case_analysis(ship, move) == 'MOVE':
                        candidate_move.append(move)
            ship.next_action = self.SHIP_ACTION_DICT[random.choice(candidate_move)]

    # TODO: refactor cases
    def case_analysis(self, ship, move) -> str:
        """
        Check if ship can move to next_pos.

        Args:
            ship: Ship
            move: Tuple

        Returns: True if next_pos is accessible.
        """
        next_pos = unify_pos(ship.position + move, self.size)
        next_cell = self.board[next_pos]

        # Check if there's any nearby enemy ship for next_pos
        safe_condition = self.find_close_enemy(ship, dis=1, pos=next_pos) == []

        # Case 1: next_cell is empty or occupied by empty ally shipyard
        # Case 2: next_cell is occupied by enemy ship with high halite
        # Case 3: next_cell won't be occupied by another ally ship next round
        case_1 = next_cell.ship is None and (next_cell.shipyard is None or next_cell.shipyard.player == self.me)
        case_2 = next_cell.ship is not None and next_cell.ship.player != self.me and next_cell.ship.halite > ship.halite
        case_3 = next_cell.position not in self.ship_next_pos
        move_condition = (case_1 or case_2) and case_3

        # Case 4: next_cell is occupied by ally ship
        # Case 5: next_cell is occupied by enemy ship with low halite
        # Case 6: next_cell is occupied by enemy shipyard
        case_4 = next_cell.ship is not None and next_cell.ship.player == self.me
        case_5 = next_cell.ship is not None and next_cell.ship.player != self.me and next_cell.ship.halite <= ship.halite
        case_6 = next_cell.shipyard and next_cell.shipyard.player != self.me

        if safe_condition and move_condition:
            return 'MOVE'
        else:
            if not safe_condition or (not case_3) or case_4:
                return 'WAIT'
            elif case_5 or case_6:
                return 'DETOUR'

        print(safe_condition, case_1, case_2, case_3, case_4, case_5, case_6)
        raise AssertionError('Unconsidered case')

    def course_reversal(self, ship: Ship):
        """
        Command function for DEPOSIT ship navigation.
        """
        if self.me.shipyards:
            nearest_shipyard = min(self.me.shipyards, key=lambda x: cal_dis(ship.position, x.position))
        else:
            shipyards = [ship.position for ship in self.me.ships if self.ship_state.get(ship.id) == 'CONVERT']
            nearest_shipyard = min(shipyards, key=lambda x: cal_dis(ship.position, x.position))
        self.navigate(ship, nearest_shipyard.position)

    def find_close_enemy(self, ship: Ship, dis: int = 1, pos: Point = None) -> list:
        """
        Find dangerous enemy ship in given distance.
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
            if 0 < cal_dis(pos, enemy_pos) <= dis and ship.halite >= enemy_ship.halite:
                close_enemy.append(enemy_pos)
        return close_enemy

    def explore_command(self, ship: Ship, radar: dict, deposit_halite: int = 500, security_dis: int = 1):
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
                self.ship_command(ship, radar['dis'] + 1, deposit_halite, security_dis)
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
            else:
                if ship.halite >= deposit_halite:
                    self.course_reversal(ship)
                else:
                    if not self.find_close_enemy(ship, security_dis):
                        self.ship_state[ship.id] = 'EXPLORE'
                        self.ship_command(ship, radar_dis, deposit_halite, security_dis)
                    else:
                        # Enemy is close, stick to DEPOSIT ship state.
                        self.course_reversal(ship)

        # EXPLORE
        elif self.ship_state[ship.id] == 'EXPLORE':
            if ship.halite >= deposit_halite:
                self.ship_state[ship.id] = 'DEPOSIT'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                self.explore_command(ship, radar, deposit_halite, security_dis)

        # COLLECT
        # Strategy 1: if ship halite reaches deposit_halite, turn COLLECT to DEPOSIT
        # Strategy 2: if enemy ship shows in radar, turn COLLECT TO DEPOSIT
        elif self.ship_state[ship.id] == 'COLLECT':
            if ship.halite >= deposit_halite:
                self.ship_state[ship.id] = 'DEPOSIT'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                if not self.find_close_enemy(ship, security_dis):
                    self.explore_command(ship, radar, deposit_halite, security_dis)
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
            if shipyard.position not in self.ship_next_pos:
                shipyard.next_action = ShipyardAction.SPAWN
                new_ship += 1
                # Add new ship position into self.ship_next_pos
                self.ship_next_pos.add(shipyard.position)

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

    def update_ship_pos(self, ship):
        """
        Update self.ship_next_pos for ship position in next round.
        """
        pos = ship.position
        if ship.next_action in {ShipAction.NORTH, ShipAction.SOUTH, ShipAction.WEST, ShipAction.EAST}:
            if ship.next_action == ShipAction.NORTH:
                next_pos = unify_pos(pos + (0, 1), self.size)
            elif ship.next_action == ShipAction.SOUTH:
                next_pos = unify_pos(pos + (0, -1), self.size)
            elif ship.next_action == ShipAction.WEST:
                next_pos = unify_pos(pos + (1, 0), self.size)
            else:
                next_pos = unify_pos(pos + (-1, 0), self.size)
            self.ship_next_pos.add(next_pos)
        elif ship.next_action is None:
            self.ship_next_pos.add(pos)

    def play(self, radar_dis=2, deposit_halite=500, security_dis=1, max_ship=5):
        """
        Main Function
        """
        print('MY TURN {}'.format(self.board.observation['step']))

        # Reset self.ship_next_pos
        self.ship_next_pos = set()

        self.convert_command()
        print('- convert command ')
        for ship in self.me.ships:
            print('-- command {}'.format(ship.id))
            self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            self.update_ship_pos(ship)
            print('---- ship state: {}'.format(self.ship_state[ship.id]))
            print('---- ship next action: {}'.format(ship.next_action))
            print('---- ship halite: {}'.format(ship.halite))

        self.spawn_command(max_ship)
        print('- spawn command')

        return self.me.next_actions
