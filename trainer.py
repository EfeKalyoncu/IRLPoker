import numpy as np
from game.gameplay_loop import PokerGame
from replay_buffer import ReplayBuffer
import random
from actor import Actor
from critic import Critic
import torch
import torch.nn.functional as F
import utils
import copy

GLOBAL_STEPS = 10
TRAINING_STEPS = 10000
EVALUATION_HANDS = 100

class PokerTrainer:
    def __init__(self, num_players=4, batch_size=2, lr = 0.01, device = "cpu"):
        #initialization for the game
        self.num_players = num_players
        self.game = PokerGame(num_players=num_players)
        self.buffer = ReplayBuffer()
        self.lr = lr
        self.device = device

        #train loop datafields instantiations
        self.batch_size = batch_size
        self.train_steps = 0
        self.eval_hands = 0
        self.global_step = 0
        self.total_eval_rewards = 0 

        #Actor parameter instantiation
        self.stddev = 15 
        self.hidden_dim = 512 #experiement with this value
        self.action_shape = 1
        self.actor = Actor( repr_dim= len(self.game.get_vectorized_state()), action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.adversary = Actor( repr_dim= len(self.game.get_vectorized_state()), action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        #Critic parameter instantion
        self.critic = Critic(repr_dim = len(self.game.get_vectorized_state()),action_shape= self.action_shape, hidden_dim=self.hidden_dim)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        print("PokerTrainer Initialized\n")
        
    def choose_action(self, agent, state):
        #get an action from our actor
        state = torch.Tensor(state)
        dist = agent(state, self.stddev)
        action = dist.sample()
        return action.detach().numpy() if isinstance(action, torch.Tensor) else action
    
    def eval_action(self,agent,state):
        state = torch.Tensor(state)
        dist = agent(state, self.stddev)
        action = dist.mean
        return action.detach().numpy() if isinstance(action, torch.Tensor) else action
    
    def eval(self):
        self.total_eval_rewards = 0
        self.eval_batch = []
        self.game = PokerGame(num_players=self.num_players)   
        while self.eval_hands < EVALUATION_HANDS:
            #start game                       
            #Play
            if self.game.action_position == 0: #if position indicates current model
                action = self.eval_action(self.actor, self.game.get_vectorized_state())[0]
                done, self.eval_batch = self.game.execute_action(action)

            else: #else adversary model plays
                action = self.eval_action(self.adversary, self.game.get_vectorized_state())[0]
                done, self.eval_batch = self.game.execute_action(action)
            
            if self.eval_batch != []:
                #when the batch is not empty accumulate rewards
                self.eval_hands += 1
                for state, action, reward in self.eval_batch:
                    if state[9] == 0:
                        self.total_eval_rewards += reward[0]
                        break
                # print(self.total_eval_rewards)

            if done != 0:
                self.game = PokerGame(num_players=self.num_players)   
        
        self.eval_progress()
        self.eval_hands = 0
        
            

    def train(self):
        #total number of training steps 
        while self.global_step < GLOBAL_STEPS:
            #start a game
            self.game = PokerGame(num_players=self.num_players)
            #get the starting state
            state = self.game.get_vectorized_state()
            
            #for 10000 or what ever is the most recent game end...
            while self.train_steps < TRAINING_STEPS:
                action = self.choose_action(self.actor, state)[0]
                # print(f'action: {action}')
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
            
            
            self.train_steps = 0
            self.global_step += 1

            #eval current and old model, then old model becomes new model
            self.eval()
            #update the agent
            if self.global_step % 50 == 0:
                self.adversary.load_state_dict(copy.deepcopy(self.actor.state_dict()))
            self.update()
            
            # Logging 
            self.log_progress()
            


    def update_actor(self, states):
        dist = self.actor(states, self.stddev)
        actions = dist.sample()
        Q, _ = self.critic(states, actions)

        actor_loss = -torch.mean(Q)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        
    def update_critic(self, states, actions, rewards):
        Q , _= self.critic(states, actions)
        critic_loss = F.mse_loss(Q, rewards[:,:1])
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
 

    def update(self):
        #get sample from replay_buffer
        for i in range(10000):
            states, actions, rewards, dones = utils.to_torch(self.buffer.sample(self.batch_size), self.device)
            states, actions, rewards, dones = states.float(), actions.float(), rewards.float(), dones
            #update critic
            self.update_critic(states, actions, rewards)

            #update actor
            self.update_actor(states)

    def log_progress(self):
        print(f"Step: {self.global_step}, Buffer Size: {len(self.buffer)}\n\n\n")

    def eval_progress(self):
        print(f"Step: {self.global_step}, Eval Rewards: {self.total_eval_rewards}")

# Main execution
trainer = PokerTrainer()
trainer.train()
