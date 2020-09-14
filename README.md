# Halite-IV
Halite Game Competition by Two Sigma

# Links
- [Competition Overview](https://www.kaggle.com/c/halite/overview/description)
- [Getting Started With Halite Notebook](https://www.kaggle.com/alexisbcook/getting-started-with-halite)
- [Halite Template Notebook](https://www.kaggle.com/mylesoneill/halite-template-bot)
- [Halite SDK Overview](https://www.kaggle.com/sam/halite-sdk-overview)
- [full source code for the SDK](https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/helpers.py)
- [Reinforcement Learning Tutorial](https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/)
- [Pytorch Documentation](http://pytorch123.com/SeventhSection/ReinforcementLearning/)

# Game Overview 
## Element
- The game is played in a 21 by 21 gridworld and lasts 400 turns. Each player starts the game with **5,000 halite and one ship**.
- Grid locations with **halite** are indicated by a light blue icon, where larger icons indicate more available halite. Halite is distributed randomly (but symmetrically) at the start of the game
- Players use **ships** to navigate the world and collect halite. A ship can only collect halite from its current position. When a ship decides to collect halite, it collects **25%** of the halite available in its cell. This collected halite is added to the ship's "cargo".
- Halite in ship cargo is not counted towards final scores. In order for halite to be counted, ships need to deposit their cargo into a **shipyard** of the same color. A ship can deposit all of its cargo in a single timestep simply by navigating to a cell containing a shipyard.

## Rule
- Players start the game with no shipyards. To get a shipyard, a player must convert a ship into a shipyard, which costs **500** halite. Also, shipyards can spawn (or create) new ships, which deducts **500** halite (per ship) from the player.
- Two ships cannot successfully inhabit the same cell. This event results in a collision, where:
    -  The smallest ship (that is, the one with the least halite in its storage) survives and steals all of the halite from other ships, all other ships will be destroyed.
    -  If an enemy ship collides with a shipyard both are destroyed.
    
## Initialization
- The game board is 21x21 cells large. The southwest (bottom-left) corner of the board is designated `(0,0)` and the northeast (top-right) corner of the board is designated `(20,20)`. The starting player positions are at `(5,5)`, `(15,5)`, `(5,15)`, and `(15,15)`. In the game code positions are given as serialized list where each cell is an integer calculated by the formula: `position = row * 21 + column`.


## Note
- The `Board` doesn't have edge. If a ship is in the side or corner, next position can be in the opposite side, e.g. in a `5 * 5` size `Board`, `ship.position = (1, 0), ship.next_action = ShipAction.South` then `ship.position = (1, 4)` in next turn.

# Game Strategy
## Bronze Bot
- **Ship Strategy**:
    - `EXPLORE`: The only mission is to get to the `max halite position`. `ship.next_action` is chosen from
     `[ShipAction.NORTH, ShipAction.SOUTH, ShipAction.EAST, ShipAction.WEST]`. Once ship arrives, turn ship state to
     `COLLECT`. 
    - `COLLECT`: `ship.next_action = None` until `ship.halite` reaches threshold value or detect enemy ships nearby.
    - `DEPOSIT`: Navigate ship to nearest `shipyard`. If `ship.halite < threshold` and confirm safety, then turn ship
     state to `EXPLORE`.

![alt text](https://app.lucidchart.com/publicSegments/view/4d9c59d5-1f32-4c9e-9afc-5ae6efff226e/image.png)

- **Ship Move Strategy**:
    - `MOVE`: If there's any available direct way to destination, randomly choose one from `candicate_move` list.
    - `DETOUR`: If there's any danger in the direct way, call `self.make_detour(ship, wait_prob=0)` to let ship move in
     another direction immediately.
    - `WAIT`: If there's unavailable but not dangerous case in the direct way. Call `self.make_detour(ship, wait_prob
    =0.5)` so that ship has 0.5 & 0.5 probability to wait and detour.
    
- **Shipyard Strategy**:
    - `CONVERT`: For `BronzeBot`, there's only one shipyard. Ship at lowest halite cell will convert if the shipyard
     is destroyed.
    - `SPAWN`: Given a maximum ship number, the shipyard will continuous spawn ships.
    
## Silver Bot

The main structure of `SilverBot` is similar with `BronzeBot` but there're much more useful features.

### New Strategy

- `Attack`: ship will tend to attack enemy ship with high halite.
- `Convert`: besides base convert strategy(always keep at least one shipyard), ship will convert to shipyard when
 find rich halite area and when ship is around by enemy.
- `Spawn`: dynamically spawn new ships according to current game situation.
- `Explore`: ship will more frequently expand radar scan distance to find higher halite cell.
- `Deposit`: if ship's halite is larger than `deposit_halite`(dynamic), then return to shipyard.
- `End`: by the end of the game, call all ships back to shipyard so that all halite can be gathered.

### What's new in code

- `radar`:
    - When scanning for `free_halite`, all enemy ships carrying higher halite will be considered as `free_halite` as
     well. So that when ship doing `EXPLORE`, it can attack enemy ship.
- `navigate`:
    - Add new parameter `detour: bool` to allow ship move to destination directly with out detour. This is designed
     for new method `final_deposit()`.
- `course_reversal`:
    - Same with `navigate`.
- `explore_command`:
    - If ship radar area free halite is rich, and there's not ally shipyard nearby then `CONVERT`.
- `spawn_command`:
    - Sort all shipyards by halite sum distributed around so that new ships can be spawned at rich halite area first.
    - Dynamically control the number of ships to spawn according to current game turn, player's halite, and rank.
- `final_deposit`:
    - When the game is about to end, call all ships directly back to shipyards to gather halite as much as possible.

### Game Simulation
