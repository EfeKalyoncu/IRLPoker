# Inverse Reinforcement Learning for Poker

## Introduction

### Motivation

Reinforcement learning has been successfully applied to many games that have consistent behaviour where the information is complete. It has also been applied to control problems without full information of the environment, however something it tends to not be good at is figuring out rewards that do not directly impact the reward, however could impact the reward.

This concept is similar to the incentives given in Montezumas Revenge to handle spaces with sparse rewards. In games of incomplete information however, there can be intrinsic information, which if learned would significantly increase the performance of the agent. As a simple example, if the agent could somehow peek at its opponents cards in a game like poker, it would be able to give much better decisions.

However loss of the main reward makes its very hard for the algorithms to learn secondary objective functions. Therefore there can be value in having the predictions be multimodal, and optimize the network not just around the primary objective function, but also secondary functions like information. However it is not clear how the primary and secondary objective functions should bbe weighted when training the network, if secondary function is given too much weight, the agent can just learn a bad policy where it always tries to get information regardless of its cost.

A way of figuring out what we should be using as a reward function for these algorithms can be to look at expert data, and comparing the rewards of agents actions, and trying to push the agent to give decisions that maximizes the same reward that would maximize the experts actions.

## Implementation

### Proposed Algorithm

1. Create the environment that executes the actions and returns state action reward triplets.
2. Initialize the Reward function to a naive reward function
3. Train an Agent using Actor Critic, save the Critic
4. Feed the states where the Expert has chosen actions to the Actor, and sample actions.
5. Calculate the corresponding predicted reward values using the critic.
6. Generate a new Reward function that maximizes $\sum_{s \in S} R(E(s)) - R(A(s))$ where E(s) is the Expert action for the state and A(s) is the actor action for the state.
7. If the newly generated Reward function is not close to the old reward function, go back to step 2 and reiterate.

### Intuition

The idea of the algorithm is that the reward for a state is dependent on the environment. And in a game with multiple players, the agents that emulate the other players are inherently a part of the environment. Therefore if we change the reward function and retrain the agents based on the new reward function, the reward predicting critic needs to be retrained as well considering the changes in the actors. 

### Implementation of Reward Function Calculator

To calculate the new reward function we take dataset of expert actions and extract the rewards associated with them. 

We then take the agent and sample 25 actions from the agent in the state expert took its action. So for each expert state action pair, we get 25 correspoding state action pairs from the agent. 

We then figure out the rewards for each action of the agent using the critic model.

We pair each reward that the agent got, with the reward of the expert, and set up Advantage of the expert Adv = a * ($ExpReward^2$ - $AgentReward^2$).

Lastly we do linear programming to maximize Adv with the variables in vector a.