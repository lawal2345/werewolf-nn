import numpy as np
import random
from collections import defaultdict

# NN with Backpropagation

class SimpleNN:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize NN with random weights. Use He initialization -> np.sqrt(2/input_size). Prevents values from being huge
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.learning_rate = learning_rate

        # Activations for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.input = None
    
    def relu(self, x): # x is inputs
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        # Backprop for ReLU
        return(x > 0).astype(float) # turns all positive numbers into 1 and all zero or negative numbers into 0.
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        self.input = X

        # Hidden layer
        self.z1 = np.dot(X, self.W1)+ self.b1
        self.a1 = self.relu(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2
    
    def backward(self, action_taken, reward):
        """Backprop using policy gradient"""
        batch_size = self.input.shape[0]
        target = np.zeros_like(self.a2) # target distribution (one-hot encoding for chosen action)
        target[0, action_taken] = 1.0 # sets the position of the chosen action to 1, making a one-hot vector. if action_taken = 2 and target = [0, 0, 0], it becomes [0, 0, 1].

        # Policy gradient: gradient = (action_prob - target) * -reward. We want to increase prob of good 
        # actions (positive reward) and decrease prob of bad actions (negative reward)
        dL_dz2 = (self.a2 - target) * (-reward)

        dL_dW2 = np.dot(self.a1.T, dL_dz2) # W2 gradient
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) # bias2 gradient

        # Hidden layer backprop
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.relu_derivative(self.z1)

        # Gradient for W1 and b1
        dL_dW1 = np.dot(self.input.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
            
        # Update weights using gradient descent
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
    
    def predict(self, X):
        """Get action probabilities"""
        return self.forward(X)


# Players in the game 
class Player:
    """Represents a single player in the game with neural network brain"""
    def __init__(self, player_id, role, num_players, learning_rate=0.01):
        self.id = player_id
        self.role = role # villager or werewolf
        self.alive = True

        # NN brain
        input_size = num_players + 3
        self.brain = SimpleNN(input_size, hidden_size=12, output_size=num_players, 
                             learning_rate=learning_rate)
        
        # Memory: store decisions made during game for training
        self.memory = []
    
    def remember(self, state, action):
        self.memory.append((state, action)) # store decision for later training
    
    def clear_memory(self):
        self.memory = [] # clear memory after training
    
    def train(self, reward):
        """
        Train on all remembered decisions with given reward
        reward: +1 for win, -1 for loss
        """
        for state, action in self.memory:
            self.brain.backward(action, reward)
        self.clear_memory()
    
    def __repr__(self):
        status = "Alive" if self.alive else "Dead"
        return f"Player {self.id} ({self.role}) - {status}"

# Werewolf game
class WerewolfGame:
    def __init__(self, num_players=6, num_werewolves=2, learning_rate=0.01, verbose=True):
        self.num_players = num_players
        self.num_werewolves = num_werewolves
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.players = []
        self.round_num = 0
        self.game_history = []
        
        self._setup_game()
    
    def _setup_game(self):
        """Assign roles randomly"""
        roles = ['werewolf'] * self.num_werewolves + \
                ['villager'] * (self.num_players - self.num_werewolves)
        random.shuffle(roles)
        
        for i in range(self.num_players):
            player = Player(i, roles[i], self.num_players, self.learning_rate)
            self.players.append(player)
        
        if self.verbose:
            print(f"\nGame has started")
            print(f"Players: {self.num_players}, Werewolves: {self.num_werewolves}, "
                  f"Villagers: {self.num_players - self.num_werewolves}")
    
    def get_alive_players(self):
        return [p for p in self.players if p.alive]
    
    def get_alive_werewolves(self):
        return [p for p in self.players if p.alive and p.role == 'werewolf']
    
    def get_alive_villagers(self):
        return [p for p in self.players if p.alive and p.role == 'villager']
    
    def encode_game_state(self, player):
        """Convert game state to numerical features for input into the NN"""
        features = []
        
        # Player alive status (1 or 0 for each player)
        for p in self.players:
            features.append(1.0 if p.alive else 0.0)
        
        # Additional context -> this is the 3 extra features that go into the NN's brain
        features.append(len(self.get_alive_werewolves()) / self.num_players)
        features.append(len(self.get_alive_villagers()) / self.num_players)
        features.append(min(self.round_num / 10.0, 1.0))  # normalized round
        
        return np.array(features).reshape(1, -1)
    
    def check_game_over(self):
        alive_werewolves = len(self.get_alive_werewolves())
        alive_villagers = len(self.get_alive_villagers())
        
        if alive_werewolves == 0:
            return ('villagers', 'All werewolves eliminated')
        if alive_werewolves >= alive_villagers:
            return ('werewolves', 'Werewolves equal or outnumber villagers')
        return (None, None)
    
    def day_phase(self):
        """Day phase: voting with NN brain"""
        if self.verbose:
            print(f"\nDAY {self.round_num}")
        
        alive_players = self.get_alive_players()
        if len(alive_players) <= 1:
            return None
        
        votes = {}
        for player in alive_players:
            state = self.encode_game_state(player)
            vote_probs = player.brain.predict(state)[0]
            
            # Mask invalid targets
            valid_votes = vote_probs.copy()
            for p in self.players:
                if not p.alive or p.id == player.id:
                    valid_votes[p.id] = 0
            
            if valid_votes.sum() == 0:
                continue
            
            valid_votes = valid_votes / valid_votes.sum()
            target_id = np.random.choice(len(valid_votes), p=valid_votes)
            
            # Remember this decision for training
            player.remember(state, target_id)
            
            if target_id not in votes:
                votes[target_id] = 0
            votes[target_id] += 1
            
            if self.verbose:
                print(f"Player {player.id} votes for Player {target_id}")
        
        if votes:
            eliminated_id = max(votes, key=votes.get)
            eliminated_player = self.players[eliminated_id]
            eliminated_player.alive = False
            
            if self.verbose:
                print(f">>> Player {eliminated_id} voted out ({eliminated_player.role})")
            
            return eliminated_player
        return None
    
    def night_phase(self):
        """Night phase: werewolf kill"""
        if self.verbose:
            print(f"\nNIGHT {self.round_num}")
        
        werewolves = self.get_alive_werewolves()
        villagers = self.get_alive_villagers()
        
        if len(werewolves) == 0 or len(villagers) == 0:
            return None
        
        # Use first werewolf's decision
        decision_maker = werewolves[0]
        state = self.encode_game_state(decision_maker)
        kill_probs = decision_maker.brain.predict(state)[0]
        
        # Mask invalid targets
        valid_targets = kill_probs.copy()
        for p in self.players:
            if not p.alive or p.role == 'werewolf':
                valid_targets[p.id] = 0
        
        if valid_targets.sum() > 0:
            valid_targets = valid_targets / valid_targets.sum()
            target_id = np.random.choice(len(valid_targets), p=valid_targets)
            target = self.players[target_id]
            
            # Remember this decision
            decision_maker.remember(state, target_id)
        else:
            target = random.choice(villagers)
        
        target.alive = False
        
        if self.verbose:
            print(f"Player {target.id} killed by werewolves ({target.role})")
        
        return target
    
    def play_game(self):
        """Main game loop"""
        while True:
            self.round_num += 1
            
            self.day_phase()
            winner, reason = self.check_game_over()
            if winner:
                if self.verbose:
                    print(f"{winner.upper()} WIN! ({reason})")
                return winner
            
            self.night_phase()
            winner, reason = self.check_game_over()
            if winner:
                if self.verbose:
                    print(f"{winner.upper()} WIN! ({reason})")
                return winner
    
    def train_players(self, winner):
        """
        Train all players based on game outcome
        Winners get positive reward, losers get negative
        """
        for player in self.players:
            if player.role == 'villager':
                reward = 1.0 if winner == 'villagers' else -1.0
            else:  # werewolf
                reward = 1.0 if winner == 'werewolves' else -1.0
            
            player.train(reward)

# Training over multiple games
class Trainer:
    """Handles training over multiple games"""
    
    def __init__(self, num_players=6, num_werewolves=2, learning_rate=0.01):
        self.num_players = num_players
        self.num_werewolves = num_werewolves
        self.learning_rate = learning_rate
        
        # Create persistent players that learn across games
        self.players = self._create_players()
        
        # Statistics
        self.stats = {
            'villager_wins': [],
            'werewolf_wins': [],
            'rounds_per_game': []
        }
    
    def _create_players(self):
        """Create players that persist across games"""
        roles = ['werewolf'] * self.num_werewolves + \
                ['villager'] * (self.num_players - self.num_werewolves)
        
        players = []
        for i in range(self.num_players):
            # Fixed roles for consistent learning
            role = roles[i]
            player = Player(i, role, self.num_players, self.learning_rate)
            players.append(player)
        
        return players
    
    def reset_game_state(self):
        """Reset player alive status for new game"""
        for player in self.players:
            player.alive = True
            player.clear_memory()
    
    def train(self, num_games=100, print_every=10):
        """Train agents over multiple games"""
        print(f"TRAINING: {num_games} games")
        
        villager_win_rate = []
        
        for game_num in range(num_games):
            self.reset_game_state()
            
            # Create game with existing players
            game = WerewolfGame(self.num_players, self.num_werewolves, 
                              self.learning_rate, verbose=False)
            game.players = self.players  # Use persistent players
            
            # Play game
            winner = game.play_game()
            
            # Train all players based on outcome
            game.train_players(winner)
            
            # Track statistics
            v_win = 1 if winner == 'villagers' else 0
            w_win = 1 if winner == 'werewolves' else 0
            
            self.stats['villager_wins'].append(v_win)
            self.stats['werewolf_wins'].append(w_win)
            self.stats['rounds_per_game'].append(game.round_num)
            
            # Calculate rolling win rate (last 20 games)
            window = 20
            if len(self.stats['villager_wins']) >= window:
                recent_v_wins = sum(self.stats['villager_wins'][-window:])
                villager_win_rate.append(recent_v_wins / window)
            
            # Print progress
            if (game_num + 1) % print_every == 0:
                recent_v = sum(self.stats['villager_wins'][-print_every:])
                recent_w = sum(self.stats['werewolf_wins'][-print_every:])
                avg_rounds = np.mean(self.stats['rounds_per_game'][-print_every:])
                
                print(f"Games {game_num+1-print_every+1}-{game_num+1}: "
                      f"Villagers: {recent_v}/{print_every}, "
                      f"Werewolves: {recent_w}/{print_every}, "
                      f"Avg Rounds: {avg_rounds:.1f}")
        
        self._print_summary(villager_win_rate)
        return self.stats
    
    def _print_summary(self, villager_win_rate):
        """Print training summary"""
        print("TRAINING COMPLETE")
        
        total_v = sum(self.stats['villager_wins'])
        total_w = sum(self.stats['werewolf_wins'])
        total_games = len(self.stats['villager_wins'])
        
        print(f"\nOverall Results:")
        print(f"  Villagers: {total_v}/{total_games} ({100*total_v/total_games:.1f}%)")
        print(f"  Werewolves: {total_w}/{total_games} ({100*total_w/total_games:.1f}%)")
        print(f"  Avg Rounds: {np.mean(self.stats['rounds_per_game']):.2f}")
        
        if len(villager_win_rate) > 0:
            print(f"\nLearning Progress (20-game rolling average):")
            print(f"  Initial win rate: {villager_win_rate[0]:.2%}")
            print(f"  Final win rate: {villager_win_rate[-1]:.2%}")
            print(f"  Change: {villager_win_rate[-1] - villager_win_rate[0]:+.2%}")


if __name__ == "__main__":
    print("WEREWOLF GAME WITH NEURAL NETWORK LEARNING")
    
    # First, demonstrate a single game with output
    print("\nDEMO: Single game with detailed output\n")
    game = WerewolfGame(num_players=6, num_werewolves=2, verbose=True)
    winner = game.play_game()
    
    # Now train agents over multiple games
    print("\n\nTRAINING: Learning through self-play\n")
    trainer = Trainer(num_players=6, num_werewolves=2, learning_rate=0.05)
    stats = trainer.train(num_games=200, print_every=20)
    
    # Test trained agents
    print("\nPOST-TRAINING: Testing learned strategies\n")
    trainer.reset_game_state()
    test_game = WerewolfGame(num_players=6, num_werewolves=2, verbose=True)
    test_game.players = trainer.players
    test_winner = test_game.play_game()