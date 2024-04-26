def return_highest_cards(suites, all_cards, guaranteed_cards=[]):
    highest_five_cards = []

    for card in all_cards:
        if card.suite in suites:
            if len(highest_five_cards) < 5:
                highest_five_cards.append(card)
                highest_five_cards.sort(key=lambda x: x.number, reverse=True)
            else:
                if card.number in guaranteed_cards:
                    card.number = 15 * (card.number + 1)
                if card.number > highest_five_cards[4].number:
                    highest_five_cards[4] = card
                    highest_five_cards.sort(key=lambda x: x.number, reverse=True)
                else:
                    continue
    if highest_five_cards[0].number > 15:
        for card in highest_five_cards:
            if card.number > 15:
                card.number = card.number / 15 - 1
        
    return highest_five_cards

def judge_high_card(hand_one, hand_two):
    for i in range(5):
        if hand_one[i].number > hand_two[i].number:
            return 0
        if hand_one[i].number < hand_two[i].number:
            return 1
    return -1

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
    exists_triple = -1

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
    
    return 0

def is_pair(all_cards):
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for card in all_cards:
        numbers[card.number] += 1
    
    pair = 0

    for i in range(13):
        if numbers[i] == 2:
            pair = i + 1
    
    return pair


def compare_hands(hand_one, hand_two):
    if is_quads(hand_one) > is_quads(hand_two):
        return 0
    elif is_quads(hand_one) > is_quads(hand_two):
        return 1
    elif is_quads(hand_one) == is_quads(hand_two) and is_quads(hand_one) > 0:
        top_five_first = return_highest_cards([0, 1, 2, 3], hand_one)
        top_five_second = return_highest_cards([0, 1, 2, 3], hand_two)
        if top_five_first[0] > top_five_second[0]:
            return 0
        elif top_five_first[0] < top_five_second[0]:
            return 1
        else:
            if top_five_first[4] > top_five_second[4]:
                return 0
            elif top_five_first[4] < top_five_second[4]:
                return 1
            else:
                return -1
    elif is_full_house(hand_one) > 0 and is_full_house(hand_two) > is_full_house(hand_one):
        return 1
    elif is_full_house(hand_one) > 0 and is_full_house(hand_two) > is_full_house(hand_one):
        return -1
    elif is_full_house(hand_one) > 0:
        return 0
    elif is_flush(hand_one) is not None and is_flush(hand_two) is not None:
        top_cards_first = return_highest_cards([is_flush(hand_one)], hand_one)
        top_cards_second = return_highest_cards([is_flush(hand_two)], hand_two)

        judge_high_card(top_cards_first, top_cards_second)
    elif is_straight(hand_one) > is_straight(hand_two):
        return 0
    elif is_straight(hand_one) > is_straight(hand_two):
        return 1
    elif is_straight(hand_one) != 0:
        return -1
    elif is_set(hand_one) > is_set(hand_two):
        return 0
    elif is_set(hand_one) < is_set(hand_two):
        return 1
    elif is_set(hand_one) != 0:
        top_cards_first = return_highest_cards([0, 1, 2, 3], hand_one, [is_set(hand_one) - 1])
        top_cards_second = return_highest_cards([0, 1, 2, 3], hand_two, [is_set(hand_one) - 1])
        judge_high_card(top_cards_first, top_cards_second)
    elif is_two_pair(hand_one) > is_two_pair(hand_two):
        return 0
    elif is_two_pair(hand_one) < is_two_pair(hand_two):
        return 1
    elif is_two_pair(hand_one) != 0:
        first_pair = is_two_pair(hand_one) % 14
        second_pair = (is_two_pair(hand_one) - first_pair) / 14
        top_cards_first = return_highest_cards([0, 1, 2, 3], hand_one, [first_pair, second_pair])
        top_cards_second = return_highest_cards([0, 1, 2, 3], hand_two, [first_pair, second_pair])
        return judge_high_card(top_cards_first, top_cards_second)
    elif is_pair(hand_one) > is_pair(hand_two):
        return 0
    elif is_pair(hand_one) < is_pair(hand_two):
        return 1
    elif is_pair(hand_one) != 0:
        top_cards_first = return_highest_cards([0, 1, 2, 3], hand_one, is_pair(hand_one) - 1)
        top_cards_second = return_highest_cards([0, 1, 2, 3], hand_one, is_pair(hand_one) - 1)
        return judge_high_card(top_cards_first, top_cards_second)
    else:
        top_cards_first = return_highest_cards([0, 1, 2, 3], hand_one)
        top_cards_second = return_highest_cards([0, 1, 2, 3], hand_one)
        return judge_high_card(top_cards_first, top_cards_second)


class card:
    def __init__(self, suite, number):
        self.suite = suite
        self.number = number


class game_state:
    def __init__(self, number_of_players):
        self.number_of_players = number_of_players