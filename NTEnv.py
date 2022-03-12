import random
import numpy as np

# 1. Deck
# ----------------------------------------------------------------------------

class Deck(object):
    """
    Deck consists of list of numbers (cards). Is initialised with standard list
    of cards in No Thanks!. Decks can be shuffled, drawn from and number of 
    cards counted.
    """
    
    def __init__(self):
        self.deck = []
        
    def build(self):
        cards_all = range(3,36)
        deck = random.sample(cards_all, 24)
        
        for card in deck:
            self.deck.append(card)
        
        # print("The deck has been shuffled.")
            
    def draw(self):
        return self.deck.pop()

    def check_end(self):
        if self.deck == []:
            return True
        
# 2. Player
# ----------------------------------------------------------------------------
        
class Player(object):
    """
    Player consists of a list of cards and number of chips in posession.
    Players can take or pass cards and/or chips. Total points to player at any
    time can be calculated.
    """
    
    def __init__(self, name):
        # Initialise the number of cards and chips
        self.name = name
        self.card_hand = list()
        self.chip_hand = 11
        
    def draw_card(self, deck, player):
        # Take a card from the deck
        global card_pool
        
        card_pool = deck.draw()
        # print(f'{self.name} draws the number ' + str(card_pool) + ".")
        
        player.weighted_play(player, deck)
    
    def take_card(self, player, deck):
        # Player takes the card
        # Card +1
        # Chips +1
        # If game continues
        global card_pool
        global chip_pool
        
        self.card_hand.append(card_pool)
        self.chip_hand += chip_pool
        
        # print(f'{self.name} takes the ' + str(card_pool) + " and " + str(chip_pool) + " chips.")
        # print(f'{self.name} has ' + str(self.chip_hand) + ' chips remaining.')
        
        chip_pool = 0
        
        if deck.check_end() != True:
            player.draw_card(deck, player)
        
    def pass_card(self):
        # Pass the card
        # Increase the number of chips on the card
        global card_pool
        global chip_pool
        
        self.chip_hand -= 1
        chip_pool += 1
        
        # print(f'{self.name} passes the ' + str(card_pool) + " and loses a chip.")
        # print(f'{self.name} has ' + str(self.chip_hand) + ' chips remaining.')
        
    def rand_play(self, player, deck):
        # random player：randomly decides to keep the card or not
        """
        Action is randomly determined. Note that players must take a card if
        they are out of chips.
        """
        global chip_pool
        decision = random.randint(0,1)
        
        
        if self.chip_hand == 0:
            decision == 0
        
        if decision == 0:
            player.take_card(player, deck)
            
        if decision == 1:
            player.pass_card()
            
    def remove_runs(player_hand):
        # Removes consecutive cards, only count the card with the smallest point in the consecutive row
        player_hand.sort()
        player_hand.reverse()
        remove_list = []
        
        for i in player_hand:
            if i-1 in player_hand:
                remove_list.append(i)

        for i in remove_list:
            player_hand.remove(i)
            
        return player_hand
    
    def point_tally(self):
        # Counting the points
        self.card_hand = Player.remove_runs(self.card_hand)
        card_points = sum(self.card_hand)
        chip_points = self.chip_hand
        return card_points - chip_points
    
    def chip_weight(chip_count):
        #return 25.5 * np.exp((-0.294) * chip_count)

        weight = (-4/55) * chip_count**2 + (-51/55) * chip_count + 20
        if weight > 0:
            return weight
        else:
            return 0

    
    def weighted_play(self, player, deck):
        # Use weights to decide if to keep the card
        global card_pool
        global chip_pool
        
        # If there is no chips left, player can only take the card.
        if self.chip_hand == 0:
            player.take_card(player, deck)
        
        take_card_hand = Player.remove_runs(self.card_hand + [card_pool])
        take_chip_hand = self.chip_hand + chip_pool
        pass_card_hand = Player.remove_runs(self.card_hand)
        pass_chip_hand = self.chip_hand - 1
        
        take_value = sum(take_card_hand) - Player.chip_weight(take_chip_hand) * take_chip_hand
        pass_value = sum(pass_card_hand) - Player.chip_weight(pass_chip_hand) * pass_chip_hand
        
        # print('take_value is '+str(take_value)+' and pass_value is '+str(pass_value))
        
        if take_value <= pass_value:
            player.take_card(player, deck)
            
        else:
            player.pass_card()

    def combine_play(self, player, deck, prob = 0.5):
        if np.random.rand() < prob:
            return self.weighted_play(player, deck)
        else:
            return self.rand_play(player, deck)
        
        
        
    
# 3. Game
# ----------------------------------------------------------------------------
class NTEnv():
    def __init__(self, num_players = 3, debug = False) -> None:
        # The first player is controlled by human player
        self.players = []
        for i in range(num_players):
            self.players.append(Player("player" + str(i)))
            # print(self.players[i].name)
        
        self.deck = Deck()
        self.deck.build()
        turn_on = np.random.randint(len(self.players))
        global card_pool
        global chip_pool
        card_pool = 0
        chip_pool = 0
        self.debug = debug

    def reset(self):
        self.deck = Deck()
        self.deck.build()
        self.turn_on = np.random.randint(len(self.players))
        global card_pool
        global chip_pool
        card_pool = 0
        chip_pool = 0
        return self.get_obs()
    
    def get_obs(self):
        # 
        state1 = np.zeros(36)
        state1[self.players[0].card_hand] = 1
        state2 = np.array(self.players[0].chip_hand).reshape(-1)

        state3 = np.zeros(36)
        for i in self.players[1:]:
            state3[i.card_hand] = 1

        global card_pool
        global chip_pool
        state4 = np.zeros(2)
        state4[0], state4[1] = card_pool, chip_pool
        # print(state1, state2, state3, state4)
        return np.concatenate([state1, state2, state3, state4])
    
    def step(self, action):
        # actio： int
        reward = 0
        done = False
        if self.turn_on != 0:
            for i in range(self.turn_on, len(self.players)):
                self.players[i].combine_play(self.players[i], self.deck)
                # self.players[i].weighted_play(self.players[i], self.deck)
                self.turn_on += 1
                self.turn_on = self.turn_on % len(self.players)
        info = {}
        cards = self.players[0].card_hand
        should_takes = [i - 1 for i in cards] + [i + 1 for i in cards]
        should_takes = list(set(should_takes))
        info['valued_cards'] = should_takes
        global card_pool
        global chip_pool
        info['card_pool'] = card_pool
        info['chip_pool'] = chip_pool
        if action == 0:
            self.players[0].take_card(self.players[0], self.deck)
            if self.deck.check_end() == True:
                done  = True
                points = [i.point_tally()  for i in self.players]
                # print(points)
                if points[0] == min(points):
                    # Win
                    reward = 100
                else:
                    reward = -100
        else:
            self.players[0].pass_card()
        
        self.turn_on += 1
        self.turn_on = self.turn_on % len(self.players)
        return self.get_obs(), reward, done, info

    
    def close(self):
        self.reset()
    def seed(self):
        self.reset()
    def render(self):
        pass
        


def Run_Game(player_1, player_2, player_3):
    """
    A game reflects an iteration of turns, until the deck emtpies and total
    points are tallied. Winner is then determined. Initialised with three
    players.
    """

    Player_1 = Player(player_1)
    Player_2 = Player(player_2)
    Player_3 = Player(player_3)

    deck = Deck()
    deck.build()
    turn_no = 1
    global card_pool
    global chip_pool 
    """
    Global used as card_pool and chip_pool need to be updated each turn so
    cannot be reset between function calls.
    """
    card_pool = 0
    chip_pool = 0
    
    Player_1.draw_card(deck, Player_1)
    
    while deck.check_end() != True:
        turn_no += 1
        
        if turn_no % 3 == 1:
            Player_1.weighted_play(Player_1, deck)
            
        if turn_no % 3 == 2:
            Player_2.weighted_play(Player_2, deck)
            
        if turn_no % 3 == 0:
            Player_3.weighted_play(Player_3, deck)
            
    else:
        P1_total = Player_1.point_tally()
        P2_total = Player_2.point_tally()
        P3_total = Player_3.point_tally()
        
        print(f'{Player_1.name} has a final score of ' + str(P1_total))
        print(f'{Player_2.name} has a final score of ' + str(P2_total))
        print(f'{Player_3.name} has a final score of ' + str(P3_total))
        
        if min(P1_total, P2_total, P3_total) == P1_total:
            print(f'{Player_1.name} has won!!!')
            
        elif min(P1_total, P2_total, P3_total) == P2_total:
             print(f'{Player_2.name} has won!!!')
         
        elif min(P1_total, P2_total, P3_total) == P3_total:
             print(f'{Player_3.name} has won!!!')
            
test = False
if test:
    # Run_Game('Alice', 'Bob', 'Claire')  
    print("----------------------Test Env----------------------")
    env = NTEnv()
    obs = env.reset()
    done = False
    while not done:
        action = np.random.choice([0,1])
        obs, r , done, _ = env.step([action])
        print()
        print(obs, r, done)
        print()
