# Werewolf Game with Neural Networks

A simulation of the Werewolf game where NN agents learn to play through self-play using neural networks implemented from scratch.

## What I Did?

I built a simulator for the game Werewolf where players are secretly assigned roles (villagers vs werewolves) and must figure out who the bad guys are through voting and elimination. Instead of humans, neural network "brains" play the game and learn strategies over time.

I built for each player a simple feedforward neural network that learns through policy gradients. After each game, winners reinforce their decisions and losers weaken theirs. Over 200 games, the NN actually get better at playing.

## How it works

**The Game:**
- 6 players: 4 villagers (good guys) and 2 werewolves (bad guys)
- Day phase: Everyone votes to eliminate a player
- Night phase: Werewolves eliminate a villager
- Villagers win if they eliminate all werewolves
- Werewolves win if they equal or outnumber villagers

**The Neural Network:**
- Input: 9 features (player alive status, faction ratios, round number)
- Hidden layer: 12 neurons with ReLU activation
- Output: 6 neurons with Softmax (vote probabilities for each player)
- Training: Policy gradient with +1 reward for wins, -1 for losses
- Backpropagation implemented from scratch (no TensorFlow/PyTorch)

**The Learning Process:**
1. Player's turn → encode game state
2. Neural network outputs vote probabilities
3. Player votes based on probabilities
4. Decision stored in memory
5. Game ends → winners get +1 reward, losers get -1
6. Backprop updates all weights
7. Repeat for 200 games → agents improve

## Example Output

```
DEMO: Single game with detailed output

Game has started
Players: 6, Werewolves: 2, Villagers: 4

DAY 1
Player 0 votes for Player 3
Player 1 votes for Player 4
Player 2 votes for Player 3
...
>>> Player 3 voted out (villager)

NIGHT 1
Player 2 killed by werewolves (villager)

...

VILLAGERS WIN! (All werewolves eliminated)
```

## Project Structure

```
werewolf_with_nn.py
├── SimpleNN          # Neural network with backprop
├── Player            # Individual player with NN brain
├── WerewolfGame      # Game mechanics and phases
└── Trainer           # Runs multiple games for training
```

## What I learned

- Implementing backpropagation from scratch really solidifies understanding
- Policy gradients are surprisingly simple: just reward good actions
- Even a basic NN can learn non-trivial game strategies
- He initialization matters way more than I expected for training stability

## Why I built this

I wanted to understand reinforcement learning fundamentals