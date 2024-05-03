# Inverse Reinforcement Learning for Poker

## Introduction

### Motivation

Reinforcement learning has been successfully applied to many games that have consistent behaviour where the environment information is complete. It has also been applied to control problems without full information of the environment, however, something it tends to not be good at is figuring out secondary objectives which are often times suboptimal in the short term but crucial to success in the future.

This motivates our decision for Poker as the game of question. Poker is a game of hidden information, each player holds cards not visible to other players. This creates an environment where decision-making must occur with incomplete data. As well, Poker requires a blend of strategic thinking, risk management, and psychological elements such as bluffing and detecting opponents' bluffs. For an RL agent, learning to balance these aspects offers a deep challenge that can push the boundaries of what artificial intelligence can achieve.

As we know from experience, one of the central challenges in applying RL to different applications, like Poker in our case, is designing a reward function that can effectively guide the agent towards not only immediate goals but also longer-term or strategic objectives. Poorly designed reward functions lead the agent to converge to a suboptimal strategy or the lack of one.

This concept is evident in Montezumas Revenge, where one of the main challenges is handling the state spaces where rewards are sparse. In games with incomplete information, there can be the presence of instrinsic information, which can theoretically significantly enhance the performance of the agent in question if learned. As a simple example, in the game of Poker, if the agent could hold information of its opponents cards in a game, it would be able to give much better decisions. So in our context, this would mean that the agent would ideally be able to develop the sense that actions which reveal more information about the opponents cards, are integral to a high success strategy. 

Training the agent primarily on the loss value of the main reward can pose challenges for algorithms and reveal difficulties in their ability to capture secondary objective functions. Therefore, adopting a multimodal approach in predictions could be beneficial, allowing the network to optimize not only for the primary objective but also for secondary functions like information gathering. However, the challenge remains in determining the appropriate weighting between these primary and secondary objectives during network training. If the secondary functions are weighted too heavily, the agent might develop a suboptimal policy, excessively prioritizing information acquisition at the expense of more critical outcomes.

Since Poker is not just about winning the current hand but also about maximizing long-term gains, this involves understanding pot odds, managing bankroll, adjusting strategies based on the progression of the game, and developing a play style that might sacrifice short-term gains for larger future rewards. Poker games, especially professional and online platforms, generate vast amounts of data about player decisions and game outcomes. This data can be utilized to train more sophisticated RL models that learn not only from their own experiences but also from the aggregated strategies of many players.

Therefore, a viable method for determining appropriate reward functions for these algorithms involves analyzing expert data. By comparing the rewards from the actions of expert players and the outcomes generated by the agent, we can align the agent's decision-making process to mimic those that maximize rewards for experts, in this case the actions of humans when playing poker. This approach ensures that the agent's strategies are optimized to reflect proven, effective tactics that also include those seemingly suboptimal for short-term rewards but prove to be effective in maximizing long-term gains. 

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

Lastly we do linear programming to maximize Adv with the variables in vector a. Then the reward function normalized based on l1 norm.

![alt text](images/RewardCalculator.png "Reward Calculations")

### Implementation of Agent with Actor-Critic

1. Initialize Actor and Critic Networks.
2. With the Actor, sample actions in the environment.
3. Save the state action reward triplets into a buffer.
4. Sample state action reward triplets from the buffer.
5. Predict the reward given state and action. Backpropogate based on the guess, and the real reward.
6. Predict an action given the state using the Actor network. Backpropogate on the action chosen based on the Q value.
7. Resample new points using the updated actor and go back to step 3.

## Issues with Implementation

### Continious action space

We need to be able to sample any action since the human actions in the dataset are continious. We cannot estimate values around their actions if we have an action space of limited size (4/5/6) like a lot of other trainers use for problems of this kind. This means that the implmentation of the environment needs to be flexible such that we can get data instead of calling everything an illegal move.

This concept creates many problems with training:

1. State space was too large. In a game of 4 players, there are a total of 13 cards that are given to players chosen from 52 cards. To make the matters worse, the state space is completely discrete, unlike visual domains where close states show similar behavior, a card being different can completely change the outcome of the game.
2. Going all in become very good strategy for early agents, because if the agent was not going all in, it could not match other agents going all in
3. If we include negative rewards to avoid illegal moves, the agent started learning if it folds all the time it can avoid negative rewards.
4. The above two conditions combined with standard deviation during the prediction caused the mean of the action to blow up either in positive or negative direction, as the agent wanted to predit 0 or all in.
5. Adding artificial constraints caused unintended albeit good strategies. In particular, explicitly banning all ins, changes the game to a forced all in game. 
6. Actor was highly reliant on the Critic, as if the critic predictions were not good, the actor would be optimizing against wrong values. The initialization of critic predicting 0 for most cases (despite the loss potentially being really high) meant that actor was thrown to a flat gradient space with bad performance very often.
7. Once the actor updates, its predictions will be very different than what the critic had previously seen. This becomes problematic for convergence reasons as the critic always is optimized based on the data from the previous actor.
8. Most trajectories are meaningless, it is hard to find good data while training.
9. Using generic Q iteration is not really possible because we care for the values we get from the critic with respect to the real rewards we observe both from the environment and the expert dataset.

### Issues with State space

As mentioned above state space was very large. This maent that it was impossible to get a meaningful sample without using a lot of resources. To combat this, we have simplified the game by removing a suit as well as removing two of the players. This meant that we could not run the algorithm we initially intended as the dataset we had for human poker hands included up to 9 seats, with a full deck, which we would need millions of samples per sampling round to have any chance of making a good critic for.

### Issues with Action space

Designing the action space was non-trivial. A portion of what the action space meant was baked into the engine we used to simulate the game. However exactly how the action space would be conveyed to the game engine would substantially change the behaviour of the underlying agent. As an example, spreading the action space such that it is hard to call all in would make it so the agent would highly value high bets, even if the hand was not good.

### Most samples containing too little information

This is another issues that we faced where the Actor would be likely to be very set on all inning because for a player with no information it is the highest expected value play. In particular, an all in would have the expected value of 0, whereas a fold would have an expected value of either -2 or -4, meaning that before the relationships of the cards are learned, only trajectory that is sampled are all ins. 

Best strategy being all in for an extended period of time put large upward pressure to the predicted action, causing the predictions to go towards infinity before any card informations could be learned.

To make things more complicated, because reward of the actions fundamentally depend on other actors actions, and in case other actors fold, the reward function for the action current agent takes will also be flat.

### Banning All ins is impossible

This was an unexpected issue that we got, however seeing it was quite magical. All in is a move in poker that guarantees that one can play in a hand even from a money disadvantage. The issue with banning all in actions in any way is that the actors quickly learned that if they at any point had money advantage, they could repeatedly make aggressive bets, and put the opposition to a situation where they could not accept the bid amount without going all in. So any agent with a money disadvantage would be forced to concede everything. Because of this starting from first hand, both actors would repeatedly bet at fisrt round. 

We have observed this style of gameplay when we played against the agent ourselves under the no all in rules, waited fora  good hand to play, just to be put in a position where we were forced to concede the hand due to the added restrictions.

### Rewards should be designed to guide gradients

As mentioned above we had to ensure that agents had some cushion to ensure the game did not become a game of chicken against the all in amount. However allowing actions to go below 0, and above the total amount of money, and considering them legal bets created rewards to be comepletely flat beyond those points, so the algorithms would not get gradients because change in the action would not affect the rewards. Therefore we needed to add punishment to values that were not within the [0-MaxMoney] range, we added negative rewards that had gradients that would guide the actor towards the correct range.

![alt text](images/RewardDistribution.png "Reward Distribution based on Action")

## Future Work

### Autoencoder

Adding an autoencoder to reduce the size of the state space, as well as normalizing it could improve the performance. Our state space was naturally very sparse because of the one hot encoded cards. Also most of the real values that were in the state vector had information relating them. Proportional differences are meant to be comparably important which an encoder would be able to capture.

### Pretraining the Critic

The training of the Actor was heavily relian on the Critic. Whereas the training of the Critic could be more independent. The main goal of the Critic is the guess the reward associated with a state, which is stochastic, however sampling of the states could be done with a uniform distribution of actions. So before we start training the Actor and use it to train the Critic, we could do supervised training with the Critic so that the starting point of the Critic network is close enough to the real reward distribution that the backpropogation of the Actor networks gradients are more accurate (since Actor network also relies on Critic netowrk).

### Decoupling Actor and Critic training

The fact that the actor and critic trained at the same time, while being highly reliant on each other meant that bad outcomes from the networks got exaggerated by each other causing divergence. To handle this the sections where the data was sampled to train actor and critic couold be decoupled, and the training of the networks could also be decoupled. So we could:

1. Sample for Critic
2. Train Critic on the sampled data
3. Sample for Actor using the Critic that is trained
4. Train the Actor using the newly sampled data
5. Go gack to 1

### Further normalization

We have normalized the action space to improve the performance, but we could further normalize the rewards and the states. The state normalization could also be handled by an autoencoder, however we could simply define rewards to be a function of the starting capital of a player in the hand, which could stabilize the training process.

### Architecture of the Actor model

For the actor model we used a simple fully connected Actor model. However if we actually wanted to make full use of a reward function that rewards different aspects of the game, having a part of the model train off of the loss of only one aspect of the reward could enable us to use more sophisticated models that make use of latent inputs.