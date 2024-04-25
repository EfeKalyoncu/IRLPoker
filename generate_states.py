import sys

f = open(sys.argv[1])

lines = f.readlines()

def char_to_suit_value(character):
    if character == 'c':
        return 0
    if character == 'd':
        return 1
    if character == 'h':
        return 2
    return 3

def card_value_to_int_value(characters):
    if characters[0] == 'A':
        return 12
    if characters[0] == 'K':
        return 11
    if characters[0] == 'Q':
        return 10
    if characters[0] == 'J':
        return 9
    else:
        return int(characters) - 2

class BoardState:
    def __init__(self):
        self.board_data_cards = [-1, -1, -1, -1, -1]

        self.money_in_pot = 0

        self.button_location = 0

        self.player_card_one = -1
        self.player_card_two = -1
        self.player_location = -1

        self.adversary_money_owned = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.adversary_money_committed = [0, 0, 0, 0, 0, 0, 0, 0, 0]

def print_board_state(board_state):
    print_list = []

    for i in range(len(board_state.board_data_cards)):
        print_list.append(board_state.board_data_cards[i])
    print_list.append(board_state.money_in_pot)

    print_list.append(board_state.button_location)
    print_list.append(board_state.player_card_one)
    print_list.append(board_state.player_card_two)
    print_list.append(board_state.player_location)

    for i in range(len(board_state.adversary_money_owned)):
        print_list.append(board_state.adversary_money_owned[i])
        print_list.append(board_state.adversary_money_committed[i])

    print(f"{print_list}")

def print_action(money_amount):
    print(f"{[money_amount]}")

def reconcile_bets(board_state):
    for i in range(len(board_state.adversary_money_committed)):
        board_state.money_in_pot += board_state.adversary_money_committed[i]
        board_state.adversary_money_committed[i] = 0

def reset_board_state(board_state):
    board_state.board_data_cards = [-1, -1, -1, -1, -1]

    board_state.money_in_pot = 0

    board_state.button_location = 0

    board_state.player_card_one = -1
    board_state.player_card_two = -1
    board_state.player_location

    board_state.adversary_money_owned = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    board_state.adversary_money_committed = [0, 0, 0, 0, 0, 0, 0, 0, 0]


board_state = BoardState()
player_map = {}
player_location = 0

for line in lines:
    if "Game started at" in line:
        reset_board_state(board_state)
        player_map.clear()
    if "is the button" in line:
        board_state.button_location = line[5]
    elif "Seat" in line:
        player_map[" ".join(line[8:].split(" ")[:-1])] = int(line[5]) - 1
        board_state.adversary_money_owned[int(line[5]) - 1] = float((line.split(" ")[-1])[1:-3])
    else:
        if "Player" in line:
            split_line = line.split(" ")
            if "(" in split_line[-1] and ")" in split_line[-1] and "blind" not in line:
                money_amount = float(split_line[-1][1:-2])
                player_no = player_map[" ".join(split_line[1:-2])]
                board_state.adversary_money_committed[player_no] += money_amount
                board_state.adversary_money_owned[player_no] -= money_amount
            elif "(" in split_line[-1] and ")" in split_line[-1]:
                money_amount = float(split_line[-1][1:-2])
                player_no = player_map[" ".join(split_line[1:-4])]
                if player_no == player_location:
                    print_board_state(board_state)
                    print_action(money_amount)
                board_state.adversary_money_committed[player_no] += money_amount
                board_state.adversary_money_owned[player_no] -= money_amount
            elif "folds" in split_line[-1]:
                player_no = player_map[" ".join(split_line[1:-1])]
                if player_no == player_location:
                    print_board_state(board_state)
                    print_action(money_amount)
            if "received card" in line:
                location = player_map[" ".join(split_line[1:-3])]
                card = split_line[-1][1:-2]
                suit_marker = card[-1]
                value_maker = card[:-1]
                suit_value = char_to_suit_value(suit_marker)
                number_value = card_value_to_int_value(value_maker)
                card_value = suit_value * 13 + number_value
                if board_state.player_card_one == -1:
                    board_state.player_card_one = card_value
                else:
                    board_state.player_card_two = card_value

        if "*** FLOP ***" in line:
            reconcile_bets(board_state)
            split_line = line.split("[")
            flop_cards = split_line[-1][:-2].split(" ")
            for i, card in enumerate(flop_cards):
                suit_marker = card[-1]
                value_maker = card[:-1]
                suit_value = char_to_suit_value(suit_marker)
                number_value = card_value_to_int_value(value_maker)
                card_value = suit_value * 13 + number_value
                board_state.board_data_cards[i] = card_value

        if "*** TURN ***" in line:
            reconcile_bets(board_state)
            turn_card = line.split(" ")[-1][1:-2]
            suit_marker = turn_card[-1]
            value_maker = turn_card[:-1]
            suit_value = char_to_suit_value(suit_marker)
            number_value = card_value_to_int_value(value_maker)
            card_value = suit_value * 13 + number_value
            board_state.board_data_cards[3] = card_value

        if "*** RIVER ***" in line:
            reconcile_bets(board_state)
            river_card = line.split(" ")[-1][1:-2]
            suit_marker = river_card[-1]
            value_maker = river_card[:-1]
            suit_value = char_to_suit_value(suit_marker)
            number_value = card_value_to_int_value(value_maker)
            card_value = suit_value * 13 + number_value
            board_state.board_data_cards[4] = card_value