import game.game_state as gs
import random
import numpy as np
import copy

class PokerGame:
    def __init__(self, num_players=9):
        self.player_capital = []
        self.player_hands = []
        self.player_pot_commitment = []
        self.cards_reserved = []
        self.player_capital_soh = []
        self.button_location = 0
        self.pot = 0
        self.action_position = 0
        self.minimum_bet = 2
        self.current_bet = 4
        self.num_players = num_players
        self.players_in_hand = 0
        self.players_agreed_on_pot = 0

        self.state_action_reward_buffer = []
        self.cards_on_board = []
        self.deck = []
        

        for suite in range (4):
            for card_number in range(13):
                self.deck.append(gs.card(suite, card_number))
        
        random.shuffle(self.deck)
        
        for i in range(num_players):
            self.player_capital.append(100)
            self.player_pot_commitment.append(0)
            self.player_hands.append(())
        
        self.init_hand()
    
    def card_value_calculate(self, card):
        return card.suite * 100 + card.number

    def find_next_action_location(self, starting_point):
        for i in range(self.num_players):
            cur_player_no = (i + starting_point + 1) % self.num_players
            if len(self.player_hands[cur_player_no]) != 0:
                return cur_player_no
        
        return -1
    
    def get_vectorized_state(self):
        state = []

        for i in range(len(self.cards_on_board)):
            state.append(self.card_value_calculate(self.cards_on_board[i]))
        for i in range(5 - len(self.cards_on_board)):
            state.append(-1)
        state.append(self.pot)

        state.append(self.button_location)
        state.append(self.card_value_calculate(self.player_hands[self.action_position][0]))
        state.append(self.card_value_calculate(self.player_hands[self.action_position][1]))
        state.append(self.action_position)

        for i in range(len(self.player_capital)):
            state.append(self.player_capital[i])
            state.append(self.player_pot_commitment[i])
            state.append(len(self.player_hands[i]) / 2)
        
        return state

    def aggregate_to_pot(self):
        for i in range(self.num_players):
            self.pot += self.player_pot_commitment[i]
            self.player_pot_commitment[i] = 0
        
        self.current_bet = 0
        self.players_agreed_on_pot = 0

    def reconcile_bets(self):
        best_hand = []
        best_player_location = []
        hands_seen = 0
        player_hand_seen = []
        for i in range(self.num_players):
            if len(self.player_hands[i]) == 0:
                player_hand_seen.append(0)
            else:
                player_hand_seen.append(2)
                hands_seen += 2
                curhand = []
                for j in range(len(self.cards_on_board)):
                    curhand.append(self.cards_on_board[j])
                curhand.append(self.player_hands[i][0])
                curhand.append(self.player_hands[i][1])

                if len(best_hand) == 0:
                    best_hand = curhand
                    best_player_location = [i]
                else:
                    comparison_result = gs.compare_hands(curhand, best_hand)
                    if comparison_result == 0:
                        best_hand = curhand
                        best_player_location = [i]
                    elif comparison_result == 1:
                        continue
                    elif comparison_result == -1:
                        best_player_location.append(i)
        
        for i in range(len(best_player_location)):
            self.player_capital[best_player_location[i]] += self.pot / len(best_player_location)
        
        for i in range(len(self.state_action_reward_buffer)):
            # The player that should receive the reward
            player_reward = self.state_action_reward_buffer[i][0][9]
            self.state_action_reward_buffer[i][2] = [self.player_capital[player_reward] - self.state_action_reward_buffer[i][2][0], hands_seen - player_hand_seen[player_reward]]

        
        
        self.pot = 0
        self.players_in_hand = 0
        self.players_agreed_on_pot = 0
        self.action_position = -1
        for i in range(5):
            self.deck.append(self.cards_on_board.pop())
        
        for i in range(self.num_players):
            if len(self.player_hands[i]) > 0:
                self.deck.append(self.player_hands[i][0])
                self.deck.append(self.player_hands[i][1])
                self.player_hands[i] = ()
        
        return self.init_hand()

    def init_hand(self):
        self.button_location = (self.button_location + 1) % 9
        self.player_capital_soh = copy.deepcopy(self.player_capital)
        for i in range(self.num_players):
            if self.player_capital[i] >= self.minimum_bet * 2:
                self.player_hands[i] = (self.deck.pop(), self.deck.pop())
                self.players_in_hand += 1
        
        for i in range(5):
            self.cards_reserved.append(self.deck.pop())
        
        action_player = -1
        blinds_put_in = 0
        for i in range(self.num_players):
            cur_player_no = (i + self.button_location + 1) % self.num_players
            if len(self.player_hands[cur_player_no]) != 0:
                if blinds_put_in == 0:
                    self.player_capital[cur_player_no] -= self.minimum_bet
                    self.player_pot_commitment[cur_player_no] += self.minimum_bet
                    action_player = cur_player_no
                    blinds_put_in += 1
                elif blinds_put_in == 1:
                    self.player_capital[cur_player_no] -= self.minimum_bet * 2
                    self.player_pot_commitment[cur_player_no] += self.minimum_bet * 2
                    blinds_put_in += 1
                    self.players_agreed_on_pot += 1
                elif blinds_put_in == 2:
                    action_player = cur_player_no
                    break
        
        if blinds_put_in < 2:
            return -1, []
        self.action_position = action_player
        self.current_bet = self.minimum_bet * 2
        returnBuffer = self.state_action_reward_buffer
        self.state_action_reward_buffer = []

        return 0, returnBuffer
        
    def get_state_for_current_player(self):
        return_list = []

        for i in range(5):
            if i >= len(self.cards_on_board):
                return_list.append(-1)
            else:
                return_list.append(self.cards_on_board[i].suite * 13 + self.cards_on_board[i].number)
        
        return_list.append(self.pot)
        return_list.append(self.button_location)
        return_list.append(self.player_hands[self.action_position][0])
        return_list.append(self.player_hands[self.action_position][1])
        return_list.append(self.action_position)

        for i in range(self.num_players):
            return_list.append(self.player_capital[i])
            return_list.append(self.player_pot_commitment[i])
    
    def execute_action(self, action):
        if action < 0:
            action = 0
        state_action_reward = [self.get_vectorized_state(), [action], [self.player_capital_soh[self.action_position], 0]]
        bet_needed = self.current_bet - self.player_pot_commitment[self.action_position]
        min_raise = self.current_bet * 2 - self.player_pot_commitment[self.action_position]
        if action >= self.player_capital[self.action_position]:
            action = self.player_capital[self.action_position]
            if action > self.current_bet:
                self.current_bet = action
                self.players_agreed_on_pot = 1
            else:
                self.players_agreed_on_pot += 1
        elif action < bet_needed:
            action = 0
            self.deck.append(self.player_hands[self.action_position][0])
            self.deck.append(self.player_hands[self.action_position][1])
            self.player_hands[self.action_position] = ()
            self.players_in_hand -= 1
        elif action < self.minimum_bet:
            action = 0
            self.players_agreed_on_pot += 1
        elif action >= bet_needed and action < min_raise:
            action = bet_needed
            self.players_agreed_on_pot += 1
        else:
            self.current_bet = self.player_pot_commitment[self.action_position] + action
            self.players_agreed_on_pot = 1
        
        state_action_reward[1] = [action]
        self.player_pot_commitment[self.action_position] += action
        self.player_capital[self.action_position] -= action
        self.state_action_reward_buffer.append(state_action_reward)
        
        if self.players_in_hand == self.players_agreed_on_pot:
            if len(self.cards_on_board) == 0:
                for i in range(3):
                    self.cards_on_board.append(self.cards_reserved.pop())
                self.action_position = self.find_next_action_location(self.button_location)
                self.aggregate_to_pot()
            elif len(self.cards_on_board) < 5:
                self.cards_on_board.append(self.cards_reserved.pop())
                self.action_position = self.find_next_action_location(self.button_location)
                self.aggregate_to_pot()
            else:
                self.aggregate_to_pot()
                return self.reconcile_bets()
        else:
            self.action_position = self.find_next_action_location(self.action_position)
        
        return 0, []
    
    def ask_action(self):
        print([str(x) for x in self.player_hands[self.action_position]])
        print([str(x) for x in self.cards_on_board])
        print(f"Position: {self.action_position}")
        print(f"Bid: {self.current_bet}")
        print(f"Currently Committed: {self.player_pot_commitment[self.action_position]}")
        print(f"Current Pot: {self.pot}")
        print(f"Current Money: {self.player_capital[self.action_position]}")
        a = float(input("--------------------------\n"))
        print(a)
        return a