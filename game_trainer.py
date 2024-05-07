import game.gameplay_loop as game_engine
import hand_autoencoder
import torch

game = game_engine.PokerGame(4)
hand_encoder = hand_autoencoder.HandAutoEncoder(torch.Tensor(game.get_vectorized_state()).shape, 512, 64)
optim = torch.optim.Adam(hand_encoder.parameters(), lr=0.000001)

counter = 0
total_loss = 0
print(len(game.deck))
while(game.execute_action(game.ask_action()) != -1):
    hand_tensor = torch.Tensor(game.get_vectorized_state()).unsqueeze(0)
    predicted = hand_encoder(hand_tensor)
    loss = torch.nn.MSELoss()(predicted, hand_tensor)
    total_loss += loss.item()
    counter += 1
    if counter % 1000 == 0:
        print(total_loss / 1000)
        total_loss = 0
    

    optim.zero_grad()
    loss.backward()
    optim.step()

    