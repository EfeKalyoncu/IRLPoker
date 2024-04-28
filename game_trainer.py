import game.gameplay_loop as game_engine

game = game_engine.PokerGame(4)

game.init_hand()
while(game.execute_action(game.ask_action()) != -1):
    print("-----------------------------")