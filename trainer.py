import numpy as np
from game.gameplay_loop import PokerGame
from replay_buffer import ReplayBuffer
import random
from actor import Actor
from critic import Critic
import torch
import torch.functional as F

class PokerTrainer:
    def __init__(self, num_players=4, buffer_capacity=10000, batch_size=32):
        #initialization for the game
        self.num_players = num_players
        self.game = PokerGame(num_players=num_players)
        self.buffer = ReplayBuffer(capacity=buffer_capacity)

        #train loop datafields instantiations
        self.batch_size = batch_size
        self.train_steps = 0
        self.eval_games = 0
        self.global_step = 0
        self.total_eval_rewards = 0 

        #Actor parameter instantiation
        self.stddev = 5 
        self.hidden_dim = 256 #experiement with this value
        self.action_shape = 1
        self.actor = Actor( repr_dim= len(self.game_vectorized_state()), action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.adversary = Actor( repr_dim= len(self.game_vectorized_state()), action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        #Critic parameter instantion
        self.critic = Critic(repr_dim = len(self.game_vectorized_states()),action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        
    def choose_action(self, agent, state):
        #get an action from our actor
        dist = agent(state, self.stddev)
        action = dist.sample()
        return action.numpy() if isinstance(action, torch.Tensor) else action
    
    def eval_action(self,agent,state):
        dist = agent(state, self.stddev)
        action = dist.mean()
        return action.numpy() if isinstance(action, torch.Tensor) else action
    
    def eval(self):
        self.total_eval_rewards = 0
        
        while self.eval_games < 100:
            #start game
            self.game = PokerGame(num_players=self.num_players)
            done = False
            while not done:                            
                #Play
                if self.game.action_position == 0: #if position indicates current model
                    action = self.eval_action(self.actor, self.get_vectorized_state())
                    done, batch = self.game.execute_action(action)
                    #when the batch is not empty accumulate rewards
                    if batch:
                        state, action, reward = batch
                        self.total_eval_rewards += reward[0]

                else: #else adversary model plays
                    action = self.eval_action(self.adversary, self.get_vectorized_state())
                    done, batch = self.game.execute_action(action)

    def train(self):
        #total number of training steps 
        for _ in self.global_step:
            #start a game
            self.game = PokerGame(num_players=self.num_players)
            #get the starting state
            state = self.game.get_vectorized_state()
            
            #for 10000 or what ever is the most recent game end...
            while self.train_steps < 10000:
                action = self.choose_action(state, self.epsilon)
                done, batch = self.game.execute_action(action)
                #if batch is not empty
                #add to the buffer
                if batch:
                    for state, action, reward in batch:
                        self.buffer.add(state, action, reward, done)                      

                if done:
                    # Start a new instance of Poker Game if only one person has the money
                    self.game = PokerGame(num_players=self.num_players)
                    state = self.game.get_vectorized_state() 
                else:
                    state = self.game.get_vectorized_state()# Update state for next iteration

                #increase the counter 
                self.train_steps += 1
            
            #eval current and old model, then old model becomes new model
            self.eval()
            #update the agent
            self.update()
            # Logging 
            self.log_progress()


    def update_actor(self, states):

        dist = self.actor(states, self.stddev)
        actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)

        Q, _ = self.critic(states, actions)

        actor_loss = -torch.mean(Q)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        
    def update_critic(self, states, actions, rewards):
        Q , _= self.critic(states, actions)

        critic_loss = F.mse_loss(Q, rewards[0])

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
 

    def update(self):
        #get sample from replay_buffer
        states, actions, rewards, dones = self.buffer.sample(self.batch_size)
        
        #update critic
        self.update_critic(states, actions, rewards)

        #update actor
        self.update_actor(states)

      

    def log_progress(self):
        print(f"Step: {self.global_step}, Buffer Size: {len(self.buffer)}")

# Main execution
trainer = PokerTrainer()
trainer.train()
