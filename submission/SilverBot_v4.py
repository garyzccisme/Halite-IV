
import random

from kaggle_environments.envs.halite.helpers import *
import numpy as np


###################
# Helper Function #
###################


def index_to_position(index: int, size: int):
    """
    Converts an index in the observation.halite list to a 2d position in the form (x, y).
    """
    y, x = divmod(index, size)
    return Point(x, (size - y - 1))


# TODO: refactor for vectorized calculation
def cal_dis(x, y):
    """
    Calculate Manhattan Distance for two points
    """
    return sum(abs(x - y))


def estimate_gain(halite, dis, t, collect_rate=0.25, regen_rate=0.02):
    """
    Calculate halite gain for given number of turn.
    """
    if dis >= t:
        return 0
    else:
        # Halite will regenerate before ship arrives
        new_halite = halite * (1 + regen_rate) ** max(0, dis - 1)
        # Ship costs (dis) rounds to arrive destination, uses (t - dis) rounds to collect halite
        return new_halite * (1 - (1 - collect_rate) ** (t - dis))


def unify_pos(pos, size):
    """
    Convert position into standard one.
    Example: Given size = 5, Point(-2, -7) -> Point(3, 3)
    """
    return pos % size


def get_shorter_move(move, size):
    """
    Given one dimension move (x or y), return the shorter move comparing with opposite move.
    The Board is actually round, ship can move to destination by any direction.
    Example: Given board size = 5, move = 3, opposite_move = -2, return -2 since abs(-2) < abs(3).
    """
    if move == 0:
        return 0
    elif move > 0:
        opposite_move = move - size
    else:
        opposite_move = move + size
    return min([move, opposite_move], key=abs)
    

#############
# Bot Class #
#############


class SilverBot:

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
            (0, -1): ShipAction.SOUTH,
            (0, 0): None,
        }

        self.global_halite_mean = 0
        # TODO: legacy
        self.halite_map = None
        self.unit_map = None

        self.unit_radar = {}
        self.radar_params = {}
        self.ship_state = {}
        self.ship_next_pos = set()
        self.ship_wait_log = {}

    def get_map(self):
        """
        In the beginning of each turn, update halite & unit map.
        """
        # Rotate the halite map so that 2-D Point can be used as an array index
        # self.halite_map = np.rot90(np.reshape(self.board.observation['halite'], (self.size, self.size)), k=3)
        self.global_halite_mean = np.mean(self.obs.halite)

        # TODO: legacy
        # Initialize unit map with all zeros
        # self.unit_map = np.zeros((self.size, self.size))
        # for i, (_, shipyards, ships) in enumerate(self.board.observation['players']):
        #     if i == self.me.id:
        #         for index in shipyards.values():
        #             self.unit_map[index_to_position(index, self.size)] += 2
        #         for index, _ in ships.values():
        #             self.unit_map[index_to_position(index, self.size)] += 1
        #     else:
        #         for index in shipyards.values():
        #             self.unit_map[index_to_position(index, self.size)] += -2
        #         for index, _ in ships.values():
        #             self.unit_map[index_to_position(index, self.size)] += -1

    # TODO: refactor for efficiency
    def radar(self, unit: Union[Ship, Shipyard], dis: int = 2):
        """
        Radar Scanning for ship & shipyard.
        Gather information of [ally, enemy, halite, free halite].
        Note: free halite is available halite here, which is estimated gain given number of turns in free area.
        Args:
            unit: Ship or shipyard
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
                scan_cell = self.board[scan_pos]
                halite[scan_pos] = scan_cell.halite

                if scan_cell.ship:
                    if scan_cell.ship.player == self.me:
                        if scan_pos != pos:
                            ally_ship.append(scan_cell.position)
                        else:
                            # scan_pos == pos, add ship current position into free_halite.
                            free_halite[scan_pos] = estimate_gain(scan_cell.halite, dis=0, t=dis + 1)
                    else:
                        enemy_ship.append(scan_cell.position)
                        # Enemy ship with rich halite is considered as free_halite.
                        if isinstance(unit, Ship) and scan_cell.ship.halite > unit.halite:
                            free_halite[scan_pos] = estimate_gain(
                                scan_cell.halite, dis=0, t=dis + 1) + scan_cell.ship.halite
                elif scan_cell.shipyard:
                    if scan_cell.shipyard.player == self.me:
                        ally_shipyard.append(scan_cell.position)
                    else:
                        enemy_shipyard.append(scan_cell.position)
                else:
                    # Cell is empty, calculate estimated halite gain for (dis + 1) turns.
                    free_halite[scan_pos] = estimate_gain(scan_cell.halite, dis=cal_dis(pos, scan_pos), t=dis + 1)

        self.unit_radar[unit.id] = {
            'dis': dis,
            'halite': halite,
            # Note: Different with BronzeBot, the value is float instead of list.
            'free_halite': free_halite,
            'ally_ship': ally_ship,
            'enemy_ship': enemy_ship,
            'ally_shipyard': ally_shipyard,
            'enemy_shipyard': enemy_shipyard,
        }

    def navigate(self, ship: Ship, des: Point, detour: bool = True):
        """
        Navigate ship to destination, give out optimal action for current turn.
        Args:
            ship: Ship.
            des: destination position.
            detour: bool, if True ship will make detour or wait.
        """
        # There are actually 4 different paths, find out the shortest one.
        move_x, move_y = unify_pos(des, self.size) - ship.position
        move_x, move_y = get_shorter_move(move_x, self.size), get_shorter_move(move_y, self.size)

        directions = [(np.sign(move_x), 0), (0, np.sign(move_y))]
        directions = [x for x in directions if x != (0, 0)]

        if detour:
            candidate_move = []
            dangerous_move = []
            wait_move = []
            for move in directions:
                pos_access = self.case_analysis(ship, move)
                if pos_access == 'MOVE':
                    candidate_move.append(move)
                elif pos_access == 'DETOUR':
                    dangerous_move.append(move)
                else:
                    wait_move.append(move)

            # Case 1: Randomly choose an action in candidate_move.
            # Case 2: Immediately make detour given dangerous_move.
            # Case 3: Add WAIT order in candidate_move, so the ship has probability to detour or wait.
            if candidate_move:
                ship.next_action = self.SHIP_ACTION_DICT[random.choice(candidate_move)]
            elif dangerous_move:
                self.make_detour(ship, dangerous_move, wait_prob=0)
            else:
                self.make_detour(ship, wait_move, wait_prob=0.5)
        else:
            # detour=False is only used at the end of the game, to gather as much as halite.
            if directions:
                ship.next_action = self.SHIP_ACTION_DICT[random.choice(directions)]

    def make_detour(self, ship, not_move_list, wait_prob: float = 0):
        """
        Strategy_1: Randomly assign the ship with an available move action excluding from not_move_list.
        Strategy_2(New): If there's no such move action and wait_prob = 0, check if ship.halite > ConvertCost.
            If so then CONVERT, else leave the ship WAIT.
        """
        candidate_move = []
        for move in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if move not in not_move_list:
                if self.case_analysis(ship, move) == 'MOVE':
                    candidate_move.append(move)
        if wait_prob == 0.5:
            candidate_move += [(0, 0)] * len(candidate_move)
        elif wait_prob != 0:
            raise ValueError('Invalid wait_prob value, only 0 or 0.5 is allowed.')

        # Randomly choose a move order for ship.
        if candidate_move:
            ship.next_action = self.SHIP_ACTION_DICT[random.choice(candidate_move)]
        # If not able to move, then CONVERT.
        elif wait_prob == 0 and ship.halite >= self.config.convertCost:
            ship.next_action = ShipAction.CONVERT
            self.ship_state[ship.id] = 'CONVERT'

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

        # Basic condition
        # Check next_pos current occupation condition
        cell_condition_1 = next_cell.ship is not None
        cell_condition_2 = next_cell.shipyard is not None

        # DETOUR
        detour_case_1 = cell_condition_1 and next_cell.ship.player != self.me and next_cell.ship.halite <= ship.halite
        detour_case_2 = cell_condition_2 and next_cell.shipyard.player != self.me
        detour_condition = detour_case_1 or detour_case_2

        # MOVE
        # Check if there's any nearby enemy ship for next_pos
        safe_condition = self.find_close_enemy(ship, dis=1, pos=next_pos) == []
        # Check if next_pos is accessible next round
        next_condition = next_cell.position not in self.ship_next_pos
        # Move cases
        move_case_1 = not cell_condition_1 and not cell_condition_2
        move_case_2 = cell_condition_1 and not cell_condition_2 and next_cell.ship.player != self.me and next_cell.ship.halite > ship.halite
        move_case_3 = not cell_condition_1 and cell_condition_2 and next_cell.shipyard.player == self.me
        move_case_4 = cell_condition_1 and next_cell.ship.next_action is not None
        move_cases = (move_case_1 or move_case_2 or move_case_3 or move_case_4)
        move_condition = move_cases and safe_condition and next_condition

        if detour_condition:
            return 'DETOUR'
        elif move_condition:
            return 'MOVE'
        else:
            return 'WAIT'

    def course_reversal(self, ship: Ship, detour: bool = True):
        """
        Command function for DEPOSIT ship navigation.
        """
        if self.me.shipyards:
            nearest_shipyard = min(self.me.shipyards, key=lambda x: cal_dis(ship.position, x.position))
        else:
            shipyards = [ship for ship in self.me.ships if self.ship_state.get(ship.id) == 'CONVERT']
            nearest_shipyard = min(shipyards, key=lambda x: cal_dis(ship.position, x.position))
        self.navigate(ship, nearest_shipyard.position, detour)

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

    def explore_command(self, ship: Ship, radar: dict, deposit_halite: int = 500,
                        security_dis: int = 1, convert_sum: float = 1000):
        """
        Command function for EXPLORE.

        Strategy_1: if ship state is EXPLORE, navigate to the position with max free halite.
        Strategy_2: if ship is in the max free halite position, turn EXPLORE to COLLECT.
        Strategy_3(New): if ship radar area free halite is rich, and there's not ally shipyard nearby then CONVERT.
        """

        # Sum up radar area halite excluding ship current cell.
        halite_sum = np.sum(list(radar['halite'].values())) - ship.cell.halite
        # Check if this area is rich and hasn't been developed (there's no shipyard in 4 distance area).
        if halite_sum >= convert_sum and all(
                [cal_dis(ship.position, shipyard.position) > 4 for shipyard in self.me.shipyards]):
            ship.next_action = ShipAction.CONVERT
            self.ship_state[ship.id] = 'CONVERT'
        else:
            max_free_halite = np.max(list(radar['free_halite'].values()))
            # Check if ship has arrived max free halite position
            if radar['free_halite'][ship.position] == max_free_halite:
                # Change ship state, ship.next_action = None
                self.ship_state[ship.id] = 'COLLECT'
            else:
                # If there's lack of halite, expand radar distance
                if max_free_halite < self.global_halite_mean and radar['dis'] <= 5:
                    self.ship_command(ship, radar['dis'] + 1, deposit_halite, security_dis, convert_sum)
                else:
                    candidate = []
                    for pos, free_halite in radar['free_halite'].items():
                        if free_halite == max_free_halite:
                            candidate.append(pos)

                    # Randomly choose a destination from candidate
                    des = random.choice(candidate)
                    self.navigate(ship, des)

    def ship_command(self, ship: Ship, radar_dis: int = 2, deposit_halite: int = 500,
                     security_dis: int = 1, convert_sum: float = 1000):
        """
        For each turn, update action of each ship.

        Args:
            ship: Ship
            radar_dis: The radar scan distance of ship.
            deposit_halite: The threshold halite value for ship to hold.
            security_dis: The distance for security check.
            convert_sum: The threshold of EXPLORE ship to CONVERT to shipyard.
        """
        # Before giving action, do radar first.
        self.radar(ship, radar_dis)
        radar = self.unit_radar[ship.id]

        # Assign EXPLORE to new ship
        if ship.id not in self.ship_state:
            self.ship_state[ship.id] = 'EXPLORE'

        # DEPOSIT
        # Strategy_1: if ship is in a shipyard, and ship.halite is 0, turn DEPOSIT to EXPLORE.
        # Strategy_2: if ship state is DEPOSIT, navigate to nearest shipyard.
        # Strategy_3: if ship halite is lower than deposit_halite and radar is clear, turn DEPOSIT to EXPLORE.
        if self.ship_state[ship.id] == 'DEPOSIT':
            # If ship has deposited halite to shipyard, assign EXPLORE to ship.
            if ship.cell.shipyard and ship.halite == 0:
                self.ship_state[ship.id] = 'EXPLORE'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                # Collect enough halite, back to shipyard.
                if ship.halite >= deposit_halite:
                    self.course_reversal(ship)
                else:
                    # Clear, ship back to EXPLORE.
                    if not self.find_close_enemy(ship, security_dis):
                        self.ship_state[ship.id] = 'EXPLORE'
                        self.ship_command(ship, radar_dis, deposit_halite, security_dis)
                    else:
                        # Enemy is nearby, stick to DEPOSIT ship state.
                        self.course_reversal(ship)

        # EXPLORE
        elif self.ship_state[ship.id] == 'EXPLORE':
            if ship.halite >= deposit_halite:
                self.ship_state[ship.id] = 'DEPOSIT'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                self.explore_command(ship, radar, deposit_halite, security_dis, convert_sum)

        # COLLECT
        # Strategy_1: if ship halite reaches deposit_halite or 10 * self.global_halite_mean, turn COLLECT to DEPOSIT
        # Strategy_2: if enemy ship shows in radar, turn COLLECT TO DEPOSIT
        elif self.ship_state[ship.id] == 'COLLECT':
            if ship.halite >= min(deposit_halite, self.global_halite_mean * 10):
                self.ship_state[ship.id] = 'DEPOSIT'
                self.ship_command(ship, radar_dis, deposit_halite, security_dis)
            else:
                if not self.find_close_enemy(ship, security_dis):
                    self.explore_command(ship, radar, deposit_halite, security_dis, convert_sum)
                else:
                    self.ship_state[ship.id] = 'DEPOSIT'
                    self.course_reversal(ship)

    def spawn_command(self, max_num_ship):
        """
        Command function for shipyard to SPAWN ship.

        Strategy_1(New): Sort empty_shipyard list so that spawning from richest shipyard.
        Strategy_2(New): Dynamically control the max_num_ship, keep me holding the most number of ships in the game.

        Args:
            max_num_ship: The upper limit of ships.
        """
        if self.me.halite > self.config.spawnCost:
            # Gather all empty shipyard and sort by radar area's free_halite sum value.
            empty_shipyard = []
            for shipyard in self.me.shipyards:
                if not shipyard.cell.ship:
                    empty_shipyard.append(shipyard)
                    self.radar(shipyard, dis=2)
            empty_shipyard.sort(key=lambda x: np.sum(
                list(self.unit_radar[x.id]['free_halite'].values())
            ))

            # Dynamically control the max_num_ship.
            # Strategy_1: in [0, 100) turn, spawn ships by given max_num_ship.
            # Strategy_2: in [100, 300) turn, spawn ships by rank.
            # Strategy_3: in [300, 400] turn, spawn ships by given max_num_ship with decay.
            if 100 <= self.obs.step < 300:
                rank = sorted(list(range(len(self.obs.players))), key=lambda x: self.obs.players[x][0], reverse=True)
                # If self.me is 1st, use lead gap to spawn more ships.
                if rank[0] == self.obs.player:
                    gap = self.me.halite - self.obs.players[rank[1]][0]
                    max_num_ship = len(self.me.ships) + gap // self.config.convertCost
                # If self.me is 2nd, first point is to ensure 2rd position.
                elif rank[1] == self.obs.player:
                    gap_1 = self.obs.players[rank[0]][0] - self.me.halite
                    gap_2 = self.me.halite - self.obs.players[rank[2]][0]
                    if gap_1 >= gap_2:
                        # Conservative strategy, 0.25 are static param.
                        max_num_ship = len(self.me.ships) + gap_2 * 0.25 // self.config.convertCost
                    else:
                        # Radical strategy, 0.75 are static param.
                        max_num_ship = len(self.me.ships) + gap_2 * 0.75 // self.config.convertCost
                else:
                    # If self.me falls behind, try best to catch up.
                    max_num_ship = max(max_num_ship, self.obs.players[rank[0]][0])
            # If game is coming to an end, stop spawning new ships aggressively.
            elif self.obs.step >= 300:
                max_num_ship *= 1 - (self.obs.step - 300) / 100

            new_ship = 0
            # Spawn Condition:
            # 1. There are available empty shipyards.
            # 2. Current and next turn ship number is lower than max_num_ship.
            # 3. Player's halite is more than Spawn Cost.
            while len(empty_shipyard) > 0 and len(self.me.ships) + new_ship < max_num_ship:
                shipyard = empty_shipyard.pop()
                shipyard.next_action = ShipyardAction.SPAWN
                new_ship += 1
                # Add new ship position into self.ship_next_pos
                self.ship_next_pos.add(shipyard.position)

    def convert_base_command(self):
        """
        Command function for ship to CONVERT to shipyard. This is base strategy to ensure there's always at least one
        shipyard.

        Strategy: if there's no shipyard, randomly pick a ship with min cell halite to convert.
        """
        if not self.me.shipyards:
            min_cell_halite = np.min([ship.cell.halite for ship in self.me.ships])
            ship_candidate = [ship for ship in self.me.ships if ship.cell.halite == min_cell_halite]
            convert_ship = random.choice(ship_candidate)
            convert_ship.next_action = ShipAction.CONVERT
            self.ship_state[convert_ship.id] = 'CONVERT'

    def update_ship_next_pos(self, ship):
        """
        Update self.ship_next_pos by ship.next_action for next turn.
        """
        pos = ship.position
        if ship.next_action is None:
            self.ship_next_pos.add(pos)
        elif ship.next_action != ShipAction.CONVERT:
            if ship.next_action == ShipAction.NORTH:
                next_pos = pos + (0, 1)
            elif ship.next_action == ShipAction.SOUTH:
                next_pos = pos + (0, -1)
            elif ship.next_action == ShipAction.WEST:
                next_pos = pos + (-1, 0)
            else:
                next_pos = pos + (1, 0)
            self.ship_next_pos.add(unify_pos(next_pos, self.size))

    def final_deposit(self):
        """
        In the last 10 turns of game, all ships stop EXPLORE & COLLECT and direct DEPOSIT.
        """
        # Directly return to shipyard
        if self.obs.step < 398:
            for ship in self.me.ships:
                self.course_reversal(ship, detour=False)
        
        # Make all ships can't be back to shipyard with enough halite CONVERT.
        elif self.obs.step == 398:
            for ship in self.me.ships:
                if not any([ship.cell.north.shipyard, ship.cell.south.shipyard,
                            ship.cell.west.shipyard, ship.cell.east.shipyard]):
                    if ship.halite >= self.config.convertCost:
                        ship.next_action = ShipAction.CONVERT

    def play(self, radar_dis=2, deposit_halite=500, security_dis=1, convert_sum: float = 1000, max_ship=5):
        """
        Main Function

        Regular flow: SPAWN -> CONVERT -> SHIP MOVE.
        Ending case: CONVERT all ships with enough halite.
        """
        # print('MY TURN {}'.format(self.board.observation['step']))
        
        self.convert_base_command()
        
        # Final strategy by the end of the game.
        if self.obs.step >= 390:
            self.final_deposit()
            
        else:
            self.get_map()
            # print('Global Mean Halite: {}'.format(self.global_halite_mean))
            self.spawn_command(max_ship)
            for ship in self.me.ships:
                # print('-- command {}'.format(ship.id))
                self.ship_command(ship, radar_dis, deposit_halite, security_dis, convert_sum)
                self.update_ship_next_pos(ship)
                # print('---- ship state: {}'.format(self.ship_state[ship.id]))
                # print('---- ship next action: {}'.format(ship.next_action))
                # print('---- ship halite: {}'.format(ship.halite))

        return self.me.next_actions
    
    
############
# Launcher #
############
    

def agent(obs,config):
    bot = SilverBot(obs,config)
    actions = bot.play(radar_dis=2, deposit_halite=300, security_dis=1, convert_sum=1500, max_ship=20)
    return actions
