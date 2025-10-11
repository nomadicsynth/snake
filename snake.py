import random
import time
import os
import argparse
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from tqdm import trange
import pickle
from positional_encoding import PositionalEncoding2D

"""
Snake RL with a compact Transformer-based DQN.

Key fixes and features:
- Correct tensor shapes for Transformer (tokens = grid cells, channel dim=3)
- Larger default grid and wall collisions (no wrap-around)
- Epsilon-greedy with replay buffer and terminal masking
- CLI to configure training/play parameters
"""

# Defaults (overridable via CLI)
DEFAULT_WIDTH = 20
DEFAULT_HEIGHT = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_GAMMA = 0.99
DEFAULT_EPS_START = 1.0
DEFAULT_EPS_END = 0.05
DEFAULT_EPS_DECAY = 0.995
DEFAULT_TARGET_UPDATE = 10
DEFAULT_NUM_EPISODES = 300
DEFAULT_HIDDEN_DIM = 64  # d_model, divisible by heads
DEFAULT_NUM_LAYERS = 2
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_LR = 3e-4
DEFAULT_REPLAY_SIZE = 50000
DEFAULT_LOG_INTERVAL = 10
DEFAULT_MAX_STEPS = 500  # per episode safety cap
DEFAULT_SEED = 42
DEFAULT_RENDER_DELAY = 0.15

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition tuple (store terminal flag for proper bootstrapping)
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class SnakeGame:
    def __init__(
        self,
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        step_penalty: float = -0.01,
        wall_collision: bool = True,
        apple_reward: float = 10.0,
        death_penalty: float = -10.0,
        num_apples: int = 1,
    ):
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.wall_collision = wall_collision
        self.apple_reward = apple_reward
        self.death_penalty = death_penalty
        # Support multiple concurrent apples. Keep backward-compatible `self.food`
        # as the current target (nearest apple), and store all apples in `self.foods`.
        self.num_apples = max(1, int(num_apples))
        self.foods = []  # list[(x,y)]
        self.food = None  # nearest apple for compatibility
        self.ate_last_step = False  # precise, per-step flag for apple consumption
        self.reset()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = (1, 0)  # Start moving right
        self.foods = []
        self._ensure_apples()
        self._update_current_food_target()
        self.score = 0
        self.game_over = False
        self.ate_last_step = False
        return self._get_state()

    def _place_food(self):
        """Place a single apple in a free cell and return its position."""
        while True:
            food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))
            if food not in self.snake and food not in self.foods:
                return food

    def _ensure_apples(self):
        """Ensure there are exactly `self.num_apples` apples on the board."""
        # Add apples if fewer than desired
        while len(self.foods) < self.num_apples:
            self.foods.append(self._place_food())
        # Remove extra apples if more than desired
        while len(self.foods) > self.num_apples:
            self.foods.pop()

    def _update_current_food_target(self):
        """Set `self.food` to the nearest apple (for backward compatibility)."""
        if not self.foods:
            self.food = None
            return
        head_x, head_y = self.snake[0]
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        self.food = min(self.foods, key=lambda f: manhattan((head_x, head_y), f))

    def _move_snake(self, direction):
        head_x, head_y = self.snake[0]
        dir_x, dir_y = direction
        new_x = head_x + dir_x
        new_y = head_y + dir_y

        # Wall collision or wrap-around
        if self.wall_collision:
            if new_x < 0 or new_x >= self.width or new_y < 0 or new_y >= self.height:
                self.game_over = True
                return float(self.death_penalty)
            new_head = (new_x, new_y)
        else:
            new_head = (new_x % self.width, new_y % self.height)

        if new_head in self.snake:
            self.game_over = True
            return float(self.death_penalty)  # Penalty for hitting itself

        self.snake.insert(0, new_head)

        reward = self.step_penalty  # small negative step to encourage progress
        self.ate_last_step = False
        # Eating logic with multiple apples
        if new_head in self.foods:
            self.score += 1
            reward = float(self.apple_reward)
            # keep growth: do not pop tail
            # remove the eaten apple and spawn a new one to maintain count
            try:
                self.foods.remove(new_head)
            except ValueError:
                pass
            self._ensure_apples()
            self.ate_last_step = True
        else:
            # normal move: pop tail
            self.snake.pop()

        # Update current target apple for compatibility
        self._update_current_food_target()

        return reward

    def _get_state(self):
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.snake:
                    state[y, x, 0] = 1  # Snake body
                elif (x, y) in self.foods:
                    state[y, x, 1] = 1  # Food (one or many)
                else:
                    state[y, x, 2] = 1  # Empty space
        return state

    def step(self, action):
        # Map action to direction
        action_to_direction = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}  # Up, Right, Down, Left
        direction = action_to_direction[action]
        reward = self._move_snake(direction)
        next_state = self._get_state()
        done = self.game_over
        return next_state, reward, done

    def _display(self):
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.snake:
                    print("O", end=" ")
                elif (x, y) in self.foods:
                    print("X", end=" ")
                else:
                    print(".", end=" ")
            print()

    # --- Runtime configuration APIs ---
    def set_num_apples(self, n: int):
        """Update the number of concurrent apples and adjust the board immediately."""
        self.num_apples = max(1, int(n))
        self._ensure_apples()
        self._update_current_food_target()


class TransformerModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        action_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        height: int,
        width: int,
    ):
        super(TransformerModel, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.input_proj = nn.Linear(in_channels, d_model)
        # Strict 2D positional encoding
        self.pos_encoder = PositionalEncoding2D(d_model, height, width)
        self._expected_seq_len = height * width
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, action_dim)

        # Optional layer norm on pooled representation
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C_in) where S = H*W, C_in=3
        returns logits: (B, action_dim)
        """
        # Project tokens
        x = self.input_proj(x)  # (B, S, d_model)
        # Validate sequence length matches H*W (supports non-square)
        assert x.size(1) == self._expected_seq_len, f"Sequence length {x.size(1)} does not match expected {self._expected_seq_len}"
        x = self.pos_encoder(x)  # (B, S, d_model)
        x = self.transformer_encoder(x)  # (B, S, d_model)
        # Mean pool over sequence
        x = x.mean(dim=1)  # (B, d_model)
        x = self.norm(x)
        logits = self.decoder(x)  # (B, action_dim)
        return logits


class DQN:
    def __init__(
        self,
        action_dim: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        lr: float,
        replay_size: int,
        height: int,
        width: int,
    ):
        # Transformer runs on tokens with in_channels=3
        self.policy_net = TransformerModel(
            in_channels=3,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            height=height,
            width=width,
        ).to(device)
        self.target_net = TransformerModel(
            in_channels=3,
            action_dim=action_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            height=height,
            width=width,
        ).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=replay_size)
        self.episode_durations = []
        self.episode_rewards = []
        self.episode_logs = []

    def state_to_tokens(self, state_np: np.ndarray) -> torch.Tensor:
        """Convert HxWx3 state to (S, C) tensor (float32)."""
        assert state_np.ndim == 3 and state_np.shape[2] == 3, "State must be HxWx3"
        h, w, c = state_np.shape
        tokens = torch.from_numpy(state_np).float().view(h * w, c)  # (S, C)
        return tokens

    def select_action(self, state_np: np.ndarray, eps: float = 0.0) -> torch.Tensor:
        # epsilon-greedy action selection
        if random.random() < eps:
            return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        tokens = self.state_to_tokens(state_np).unsqueeze(0).to(device)  # (1, S, C)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(tokens)  # (1, A)
            action = q_values.argmax(dim=1, keepdim=True).long()
        return action

    def optimize_model(self, batch_size: int, gamma: float):
        if len(self.memory) < batch_size:
            return
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        # states: list of (S,C) tensors
        state_batch = torch.stack(batch.state).to(device)  # (B, S, C)
        next_state_batch = torch.stack(batch.next_state).to(device)  # (B, S, C)
        action_batch = torch.cat(batch.action).to(device)  # (B, 1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)

        # Compute Q(s,a)
        q_values = self.policy_net(state_batch)  # (B, A)
        state_action_values = q_values.gather(1, action_batch)

        # Compute V(s') = max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_state_batch)  # (B, A)
            next_state_values = next_q.max(dim=1, keepdim=True)[0]  # (B,1)
            expected_state_action_values = reward_batch + (1.0 - done_batch) * (gamma * next_state_values)

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state_tokens: torch.Tensor, action: torch.Tensor, next_state_tokens: torch.Tensor, reward: float, done: bool):
        self.memory.append(Transition(state_tokens, action, next_state_tokens, reward, done))

    def log_episode(self, episode, states, actions, rewards, scores):
        log_entry = {"episode": episode, "states": states, "actions": actions, "rewards": rewards, "score": scores}
        self.episode_logs.append(log_entry)

    def save_logs(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.episode_logs, f)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    width: int,
    height: int,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    target_update: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    lr: float,
    replay_size: int,
    log_interval: int,
    max_steps: int,
    model_path: str,
    seed: int,
):
    set_seed(seed)
    env = SnakeGame(width=width, height=height, wall_collision=True)
    action_dim = 4
    dqn = DQN(
        action_dim=action_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        lr=lr,
        replay_size=replay_size,
        height=height,
        width=width,
    )
    eps = eps_start

    pbar = trange(num_episodes, desc="Training", unit="ep")
    for episode in pbar:
        state = env.reset()
        states_log = []
        actions_log = []
        rewards_log = []
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = dqn.select_action(state, eps)
            next_state, reward, done = env.step(action.item())

            # Store tokens in replay
            state_tokens = dqn.state_to_tokens(state)
            next_state_tokens = dqn.state_to_tokens(next_state)
            dqn.remember(state_tokens, action, next_state_tokens, float(reward), bool(done))

            # Logs
            states_log.append(state)
            actions_log.append(action.item())
            rewards_log.append(float(reward))

            # Move on
            state = next_state
            steps += 1

            # Optimize
            dqn.optimize_model(batch_size=batch_size, gamma=gamma)

        dqn.episode_rewards.append(sum(rewards_log))
        dqn.episode_durations.append(env.score)
        dqn.log_episode(episode, states_log, actions_log, rewards_log, env.score)

        if (episode + 1) % target_update == 0:
            dqn.update_target_net()

        eps = max(eps_end, eps * eps_decay)

        # Update progress bar postfix every episode
        window = min(log_interval, len(dqn.episode_rewards)) or 1
        avg_reward = float(np.mean(dqn.episode_rewards[-window:]))
        avg_score = float(np.mean(dqn.episode_durations[-window:]))
        pbar.set_postfix({
            "ep_reward": f"{sum(rewards_log):.1f}",
            "avgR": f"{avg_reward:.2f}",
            "score": env.score,
            "avgS": f"{avg_score:.2f}",
            "eps": f"{eps:.3f}",
        })

    # Save model and logs
    torch.save(dqn.policy_net.state_dict(), model_path)

    dqn.save_logs("episode_logs.pkl")
    # Plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(dqn.episode_rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.subplot(1, 2, 2)
    plt.plot(dqn.episode_durations)
    plt.title("Episode Scores")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("training_stats.png")
    plt.close()


def play(model_path: str, width: int, height: int, render_delay: float):
    env = SnakeGame(width=width, height=height, wall_collision=True)
    action_dim = 4
    dqn = DQN(
        action_dim=action_dim,
        d_model=DEFAULT_HIDDEN_DIM,
        num_layers=DEFAULT_NUM_LAYERS,
        num_heads=DEFAULT_NUM_HEADS,
        dropout=DEFAULT_DROPOUT,
        lr=DEFAULT_LR,
        replay_size=DEFAULT_REPLAY_SIZE,
        height=height,
        width=width,
    )
    dqn.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    dqn.policy_net.eval()

    state = env.reset()
    done = False

    while not done:
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Score: {env.score}")
        action = dqn.select_action(state, eps=0.0)
        next_state, reward, done = env.step(action.item())
        state = next_state
        env._display()
        time.sleep(render_delay)


def reconstruct_episode(episode_log, render_delay: float = 0.1):
    states = episode_log["states"]
    actions = episode_log["actions"]
    rewards = episode_log["rewards"]
    # Grid size inferred from first state
    h, w, _ = states[0].shape

    for i, state in enumerate(states):
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Step: {i}, Action: {actions[i]}, Reward: {rewards[i]:.2f}")
        # Render from state tensor values
        for y in range(h):
            row = []
            for x in range(w):
                cell = state[y, x]
                if cell[0] == 1:  # snake
                    row.append("O")
                elif cell[1] == 1:  # food
                    row.append("X")
                else:
                    row.append(".")
            print(" ".join(row))
        time.sleep(render_delay)


def build_arg_parser():
    p = argparse.ArgumentParser(description="Snake with Transformer DQN")
    sub = p.add_subparsers(dest="mode", required=False)

    # Train args
    pt = sub.add_parser("train", help="Train the agent")
    pt.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    pt.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    pt.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES)
    pt.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    pt.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    pt.add_argument("--eps-start", type=float, default=DEFAULT_EPS_START)
    pt.add_argument("--eps-end", type=float, default=DEFAULT_EPS_END)
    pt.add_argument("--eps-decay", type=float, default=DEFAULT_EPS_DECAY)
    pt.add_argument("--target-update", type=int, default=DEFAULT_TARGET_UPDATE)
    pt.add_argument("--d-model", type=int, default=DEFAULT_HIDDEN_DIM)
    pt.add_argument("--layers", type=int, default=DEFAULT_NUM_LAYERS)
    pt.add_argument("--heads", type=int, default=DEFAULT_NUM_HEADS)
    pt.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    pt.add_argument("--lr", type=float, default=DEFAULT_LR)
    pt.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    pt.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL)
    pt.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    pt.add_argument("--model-path", type=str, default="snake_transformer.pth")
    pt.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Play args
    pp = sub.add_parser("play", help="Play using a trained model")
    pp.add_argument("--model-path", type=str, default="snake_transformer.pth")
    pp.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    pp.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    pp.add_argument("--render-delay", type=float, default=DEFAULT_RENDER_DELAY)

    # Utility: reconstruct episode from logs
    pr = sub.add_parser("reconstruct", help="Reconstruct a logged episode")
    pr.add_argument("--logs", type=str, default="episode_logs.pkl")
    pr.add_argument("--index", type=int, default=0)
    pr.add_argument("--render-delay", type=float, default=DEFAULT_RENDER_DELAY)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "play":
        play(model_path=args.model_path, width=args.width, height=args.height, render_delay=args.render_delay)
        return
    elif args.mode == "reconstruct":
        with open(args.logs, "rb") as f:
            episode_logs = pickle.load(f)
        idx = max(0, min(args.index, len(episode_logs) - 1))
        reconstruct_episode(episode_logs[idx], render_delay=args.render_delay)
        return

    # Default to train when mode is None or "train"
    if not hasattr(args, "width"):
        # If no subcommand was given, fall back to defaults for training
        args = argparse.Namespace(
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            episodes=DEFAULT_NUM_EPISODES,
            batch_size=DEFAULT_BATCH_SIZE,
            gamma=DEFAULT_GAMMA,
            eps_start=DEFAULT_EPS_START,
            eps_end=DEFAULT_EPS_END,
            eps_decay=DEFAULT_EPS_DECAY,
            target_update=DEFAULT_TARGET_UPDATE,
            d_model=DEFAULT_HIDDEN_DIM,
            layers=DEFAULT_NUM_LAYERS,
            heads=DEFAULT_NUM_HEADS,
            dropout=DEFAULT_DROPOUT,
            lr=DEFAULT_LR,
            replay_size=DEFAULT_REPLAY_SIZE,
            log_interval=DEFAULT_LOG_INTERVAL,
            max_steps=DEFAULT_MAX_STEPS,
            model_path="snake_transformer.pth",
            seed=DEFAULT_SEED,
        )

    train(
        width=args.width,
        height=args.height,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        target_update=args.target_update,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        replay_size=args.replay_size,
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        model_path=args.model_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
