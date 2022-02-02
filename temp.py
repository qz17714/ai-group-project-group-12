import random

# 1. Deck
# ----------------------------------------------------------------------------
        
class Deck(object):
    """
    Deck consists of list of numbers (cards). Is initialised with standard list 
    of cards in No Thanks!. Decks can be shuffled, number of cards counted.
    """
    
    def __init__(self):
        self.cards = []
        self.build()
        self.shuffle()
        
    def build(self):
        cards_all = range(3,36)
        cards_disc = random.sample(cards_all, 9)
        cards_game = cards_all - cards_disc
        
        for card in cards_game:
            self.cards.append(card)
            
    def shuffle(self):
        random.shuffle(self.cards)
        
    def check_end(self):
        if self.cards == []:
            return True
 
# 2. Player
# ----------------------------------------------------------------------------
        
class Player(object):
    """
    Player consists of a list of cards and number of chips in possession. 
    Players can draw cards from deck, total points to player can be calculated.
    """
    
    def __init__(self, chips = 11):
        self.card_hand = list()
        self.chip_num = chips
        
    def draw_card(self):
        return self.cards.pop()
    
    def point_tally(self):
        self.card_hand = [int(x) for x in self.card_hand]
        self.card_hand  = self.card_hand.sort()
        for i in self.card_hand:
            if i+1 in self.card_hand:
                self.card_hand.remove(i+1)
                
        card_points = sum(self.card_hand)
        chip_points = self.chip_num
        return card_points - chip_points

# 3. Turn
# ----------------------------------------------------------------------------
        
class Turn(object):
    """
    Captures the process of a turn, that consists of:
       - Initialisation of chips
       - Chosen action of player
    """
    
    def __init__(self, deck, player):
        self.deck = deck
        self.player = player
        self.card_transfer = list()
        self.chip_transfer = 0
        self.start_up()
        
    def action(self, player):
        """
        Action is randomly determined unless only one option is available.
        """
        if self.card_transfer == 0:
            self.card_transfer.append(player.draw_card())
            
        decision = random.randint(0,1)
        
        if self.chip_num == 0:
            decision = 0
        
        if decision == 0:
            self.card_hand.append(self.card_transfer)
            self.chip_num += self.chip_transfer
            self.card_transfer = list()
            self.chip_transfer = 0
        
        if decision == 1:
            self.chip_num -= 1
            self.chip_transfer += 1  
            
# 3. Game
# ----------------------------------------------------------------------------
            
class Game(object):
    """
    A game reflects an iteration of turns, until the deck empties and total
    points are tallied. Winner is then determined . Initialised with three 
    players and a turn object.
    """
    
    def __init__(self, player_1_name, player_2_name, player_3_name, player, deck):
        self.player_1 = Player(player_1_name)
        self.player_2 = Player(player_2_name)
        self.player_3 = Player(player_3_name)
        self.turn = Turn(deck = Deck(), player_1 = self.player_1, player_2 = self.player_2, player_3 = self.player_3)
        
        self.turn_no = 0
        
        while deck.check_end(0 == False):
            self.turn_no += 1
            
            if self.turn_no%2 == 1: 
                self.turn.action(self.player_1)
                
            if self.turn_no%2 == 2: 
                self.turn.action(self.player_2)
                
            if self.turn_no%2 == 0: 
                self.turn.action(self.player_3)
                
        else:
            P1_total = player.point_tally(self.player_1)
            P2_total = player.point_tally(self.player_2)
            P3_total = player.point_tally(self.player_3)
            
            if min(P1_total, P2_total, P3_total) == P1_total:
                print("Player_1 has won!!!")
                
            elif min(P1_total, P2_total, P3_total) == P2_total:
                 print("Player_2 has won!!!")
             
            elif min(P1_total, P2_total, P3_total) == P3_total:
                 print("Player_3 has won!!!")
             
            
            