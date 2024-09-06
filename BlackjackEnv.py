import numpy as np
import numpy.random as random
import gymnasium as gym
from gymnasium.spaces import *

class BlackjackEnv(gym.Env):
    deal_helper_arr = np.arange(13)
    total_helper_arr = np.array([11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])

    def __init__(self, n_decks=6, deck_depth=0.8):
        super().__init__()

        self.n_decks = n_decks
        self.deck_depth = 0.8
        self.observation_space = Dict(
            {
                'n_cards' : Discrete(n_decks * 52),
                'cards_remaining': MultiDiscrete(np.full(13, n_decks * 4)), #calculated from all previous rounds played
                'dealer_card': Discrete(13),
                'hands': MultiDiscrete(np.full((4, 13), n_decks * 4)), #4 hands, keep track of how many of each type of card in each hand
            }
        )
        
        self.action_space = Discrete(5)

        self.reset()
    
    def reset(self, seed=None):
        random.seed(seed)

        self.n_cards = self.n_decks * 52
        self.cards_remaining = np.full(13, self.n_decks * 4)

        self.start_round()

        obs = {
            'n_cards': self.n_cards,
            'cards_remaining': self.cards_remaining,
            'dealer_card': self.dealer_card,
            'hands': self.hands
        }

        return obs, {}

    def step(self, action):
        match action:
            case 0:
                self.hit()
            case 1:
                self.stand()
            case 2:
                self.double()
            case 3:
                self.split()
            case 4:
                self.surrender()

        self.advance_hands()

        if self.cur_hand >= self.n_hands:
            term = False
            rew = self.end_round()
            if self.n_cards < (1 - self.deck_depth) * self.n_decks * 52:
                term = True
            else:
                rew += self.start_round()
            obs = {
                'n_cards': self.n_cards,
                'cards_remaining': self.cards_remaining,
                'dealer_card': self.dealer_card,
                'hands': self.hands
            }
            return obs, rew, term, False, {'round_over': True}


        obs = {
            'n_cards': self.n_cards,
            'cards_remaining': self.cards_remaining,
            'dealer_card': self.dealer_card,
            'hands': self.hands
        }
        return obs, 0, False, False, {}
    
                
    def hit(self):
        card = self.deal()
        self.hands[self.cur_hand][card] += 1
        self.hand_totals[self.cur_hand] = BlackjackEnv.add_total(self.hand_totals[self.cur_hand], card)

    def stand(self):
        self.cur_hand += 1

    def double(self):
        self.doubles[self.cur_hand] = True
        card = self.deal()
        self.hands[self.cur_hand][card] += 1
        self.hand_totals[self.cur_hand] = BlackjackEnv.add_total(self.hand_totals[self.cur_hand], card)
        self.cur_hand += 1

    def split(self):
        self.n_hands += 1

        card = np.argmax(self.hands[self.cur_hand])

        self.hands[self.cur_hand][card] = 1
        self.hand_totals[self.cur_hand] = BlackjackEnv.total_helper_arr[card]

        self.hands[self.cur_hand + 1][card] = 1
        self.hand_totals[self.cur_hand + 1] = BlackjackEnv.total_helper_arr[card]

        card = self.deal()
        self.hands[self.cur_hand][card] += 1
        self.hand_totals[self.cur_hand] = BlackjackEnv.add_total(self.hand_totals[self.cur_hand], card)
        card = self.deal()
        self.hands[self.cur_hand + 1][card] += 1
        self.hand_totals[self.cur_hand + 1] = BlackjackEnv.add_total(self.hand_totals[self.cur_hand + 1], card)

    def surrender(self):
        self.surrenders[self.cur_hand] = True
        self.cur_hand += 1

    def advance_hands(self):
        while self.cur_hand < self.n_hands:
            if self.hand_totals[self.cur_hand] >= 21:
                self.cur_hand += 1
            else:
                break

    def deal(self):
        card = random.choice(BlackjackEnv.deal_helper_arr, p=self.cards_remaining / self.n_cards)
        self.n_cards -= 1
        self.cards_remaining[card] -= 1
        return card

    @staticmethod
    def add_total(total, card):
        if card == 0:
            if total <= 10:
                return total + 11
            else:
                return total + 1
        return total + BlackjackEnv.total_helper_arr[card]

    def start_round(self):
        if self.n_cards < (1 - self.deck_depth) * self.n_decks * 52:
            return 0
        
        self.hands = np.zeros((4, 13), dtype=np.int32)
        self.hand_totals = np.zeros(4)
        self.doubles = np.zeros(4).astype(bool)
        self.surrenders = np.zeros(4).astype(bool)
        self.n_hands = 1
        self.cur_hand = 0
        self.dealer_total = 0

        card = self.deal()
        self.hands[0][card] += 1
        self.hand_totals[0] = BlackjackEnv.add_total(self.hand_totals[0], card)

        self.dealer_card = self.deal()

        card = self.deal()
        self.hands[0][card] += 1
        self.hand_totals[0] = BlackjackEnv.add_total(self.hand_totals[0], card)

        rew = 0

        if self.hand_totals[0] == 21:
            rew = self.end_round()
            rew += self.start_round()
        
        return rew


    def end_round(self):
        #functionally identical to choosing it at round start but more efficient this way
        self.dealer_hole_card = self.deal() 
        self.dealer_total = self.dealer_card + self.dealer_hole_card
        dealer_has_blackjack = self.dealer_total == 21
        while self.dealer_total < 17:
            card = self.deal()
            self.dealer_total = BlackjackEnv.add_total(self.dealer_total, card)

        rew = 0
        for hand in range(self.n_hands):
            if self.surrenders[hand]:
                rew -= 0.5
                continue

            cur_rew = 0
            if self.hand_totals[hand] > 21:
                cur_rew = -1
            elif self.hand_totals[hand] == 21:
                if np.sum(self.hands[hand]) == 2: #does player have blackjack?
                    if self.dealer_total != 21:
                        cur_rew = 1.5
                else:
                    if dealer_has_blackjack:
                        cur_rew = -1
                    elif self.dealer_total != 21:
                        cur_rew = 1
            elif self.hand_totals[hand] < 21:
                if self.dealer_total > 21:
                    cur_rew = 1
                elif self.dealer_total > self.hand_totals[hand]:
                    cur_rew = -1
                elif self.dealer_total < self.hand_totals[hand]:
                    cur_rew = 1
            
            if self.doubles[hand]:
                cur_rew *= 2

            rew += cur_rew

        return rew
    
    def get_legal(self):
        first_action = np.sum(self.hands[self.cur_hand]) == 2
        split = first_action and np.max(self.hands[self.cur_hand]) == 2 and self.n_hands < 4
        return np.array([True, True, first_action, split, first_action])