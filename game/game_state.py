def return_highest_cards(suites, all_cards):
    highest_five_cards = []

    for card in all_cards():
        if card.suite in suites:
            if len(highest_five_cards) < 5:
                highest_five_cards.append(card)
                highest_five_cards.sort(key=lambda x: x.number, reverse=True)
            else:
                if card.number > highest_five_cards[4].number:
                    highest_five_cards[4] = card
                    highest_five_cards.sort(key=lambda x: x.number, reverse=True)
                else:
                    continue
    return highest_five_cards

def is_flush(all_cards):
    suite_count = [0, 0, 0, 0]

    for card in all_cards:
        suite_count[card.suite] += 1
    
    for suite, count in enumerate(suite_count):
        if count > 4:
            return suite
    
    return None

def is_straight(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    straight_counter = 0 if numbers[12] < 1 else 1
    highest_straight = 0
    for i in range(13):
        if numbers[i] == 0:
            straight_counter = 0
        else:
            straight_counter += 1
            if straight_counter >= 5:
                highest_straight = i
    
    return highest_straight

def is_quads(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    for i in range(13):
        if numbers[i] == 4:
            return i + 1

    return 0

def is_full_house(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    exists_pair = 0
    exists_triple = 0

    for i in range(13):
        if numbers[i] == 2:
            exists_pair = i + 1
        elif numbers[i] == 3:
            if exists_triple > exists_pair:
                exists_pair = exists_triple
                exists_triple = i + 1
            else:
                exists_triple = i + 1
            
    return exists_triple * 14 + exists_pair

def is_set(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    exists_triple = 0

    for i in range(13):
        if numbers[i] == 3:
            exists_triple = i + 1

    return exists_triple

def is_two_pair(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    higher_pair = 0
    lower_pair = 0

    for i in range(13):
        if numbers[i] == 2:
            lower_pair = higher_pair
            higher_pair = i + 1
    
    if lower_pair > 0:
        return 14 * higher_pair + lower_pair

def is_pair(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    pair = 0

    for i in range(13):
        if numbers[i] == 2:
            pair = i + 1
    
    return pair


class card:
    def __init__(self, suite, number):
        self.suite = suite
        self.number = number


class game_state:
    def __init__(self, number_of_players):
        self.number_of_players = number_of_players
