import os
import math
import argparse
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import warnings
from dotenv import load_dotenv
from contextlib import nullcontext

# Load environment variables from .env file
load_dotenv()

# Suppress pydantic warnings from stable-baselines3
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import time
from collections import deque
import wandb

# Import from snake.py
from snake import SnakeGame, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_HIDDEN_DIM, DEFAULT_NUM_LAYERS, DEFAULT_NUM_HEADS, DEFAULT_DROPOUT, DEFAULT_LR, DEFAULT_REPLAY_SIZE, DEFAULT_BATCH_SIZE, DEFAULT_GAMMA, DEFAULT_EPS_START, DEFAULT_EPS_END, DEFAULT_TARGET_UPDATE, DEFAULT_NUM_EPISODES, DEFAULT_LOG_INTERVAL, DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_RENDER_DELAY, set_seed
from positional_encoding import PositionalEncoding2D
from stable_baselines3.common.env_checker import check_env

try:
    # Optional: Muon optimizer for hidden weights
    from muon import MuonWithAuxAdam
    _MUON_AVAILABLE = True
except Exception:
    _MUON_AVAILABLE = False


# Helper functions to read environment variables with fallback defaults
def get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with fallback to default."""
    val = os.getenv(key)
    return int(val) if val is not None else default


def get_env_float(key: str, default: float) -> float:
    """Get float environment variable with fallback to default."""
    val = os.getenv(key)
    return float(val) if val is not None else default


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with fallback to default."""
    val = os.getenv(key)
    if val is None:
        return default
    return val.lower() in ('true', '1', 'yes', 'on')


# Read optimization settings from .env
ENV_BATCH_SIZE = get_env_int('OPTIMAL_BATCH_SIZE', DEFAULT_BATCH_SIZE)
ENV_USE_TORCH_COMPILE = get_env_bool('USE_TORCH_COMPILE', False)
ENV_USE_FLASH_ATTENTION = get_env_bool('USE_FLASH_ATTENTION', False)
ENV_USE_BF16 = get_env_bool('USE_BF16', False)


def cosine_schedule(start: float, end: float):
    # progress_remaining: 1.0 at start -> 0.0 at end
    def _sched(progress_remaining: float) -> float:
        # cosine from start -> end
        cos = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return end + (start - end) * cos

    return _sched


# ---- Transformer blocks (batch_first) ----
class TransformerExtractor(BaseFeaturesExtractor):
    """
    Input obs: Box(low=0, high=1, shape=(H, W, 3), dtype=float32)
    Produces a features vector for SB3’s DQN.
    """
    def __init__(self, observation_space: spaces.Box, d_model=DEFAULT_HIDDEN_DIM, n_layers=DEFAULT_NUM_LAYERS, n_heads=DEFAULT_NUM_HEADS, dropout=DEFAULT_DROPOUT, features_dim=128):
        super().__init__(observation_space, features_dim)
        assert observation_space.shape[-1] == 3, "Expected HxWx3 observation"
        self.h, self.w, self.c = observation_space.shape
        self.d_model = d_model

        self.input_proj = nn.Linear(3, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pos = PositionalEncoding2D(d_model, self.h, self.w)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Accept (B, H, W, 3) or (B, 3, H, W)
        if obs.dim() != 4:
            raise ValueError(f"Expected 4D obs, got {tuple(obs.shape)}")

        if obs.shape[-1] == 3:
            # (B, H, W, 3)
            x_img = obs
        elif obs.shape[1] == 3:
            # (B, 3, H, W) -> (B, H, W, 3)
            x_img = obs.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"Expected channels==3 in dim -1 or 1, got {tuple(obs.shape)}")

        b, h, w, c = x_img.shape
        assert h == self.h and w == self.w and c == 3, f"Unexpected obs shape {tuple(x_img.shape)}, expected (*, {self.h}, {self.w}, 3)"
        tokens = x_img.reshape(b, h * w, c)          # (B, S, 3)
        x = self.input_proj(tokens)                 # (B, S, d)
        x = self.pos(x)                             # (B, S, d)
        x = self.encoder(x)                         # (B, S, d)
        x = x.mean(dim=1)                           # (B, d)
        x = self.norm(x)                            # (B, d)
        return self.head(x)                         # (B, features_dim)

# ---- Gymnasium wrapper for SnakeGame ----
class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 6}

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        wall_collision: bool = True,
        step_penalty: Optional[float] = None,
        shaping_coef: float = 0.0,
        apple_reward: Optional[float] = None,
        death_penalty: Optional[float] = None,
        num_apples: int = 1,
        render_mode: str = "ansi",
        # Loop shaping controls
        loop_penalty_coef: float = 0.0,
        loop_end_bonus: float = 0.0,
        loop_min_period: int = 4,
        loop_max_period: int = 20,
    ):
        super().__init__()
        # Allow overriding the environment's default step penalty (discourage inefficiency)
        sg_kwargs = {"width": width, "height": height, "wall_collision": wall_collision, "num_apples": int(num_apples)}
        if step_penalty is not None:
            sg_kwargs["step_penalty"] = float(step_penalty)
        if apple_reward is not None:
            sg_kwargs["apple_reward"] = float(apple_reward)
        if death_penalty is not None:
            sg_kwargs["death_penalty"] = float(death_penalty)
        self.game = SnakeGame(**sg_kwargs)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.game.height, self.game.width, 3), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._last_obs = None
        self.shaping_coef = float(shaping_coef)
        self.render_mode = render_mode
        self.np_random = None  # set by super().reset(seed=...)
        # runtime-configurable limits
        self.default_max_steps = DEFAULT_MAX_STEPS

        # Loop shaping params
        self.loop_penalty_coef = float(loop_penalty_coef)
        self.loop_end_bonus = float(loop_end_bonus)
        self.loop_min_period = int(loop_min_period)
        self.loop_max_period = int(loop_max_period)
        self._loop_max_history = max(2 * self.loop_max_period, self.loop_max_period * 10)
        # Loop tracking state
        self._action_hist = deque(maxlen=self._loop_max_history)
        self._in_loop_prev = False
        self._loop_repeats_prev = 0
        self._loop_period_prev = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            # Seed any custom RNGs and the action space
            set_seed(seed)
            self.action_space.seed(seed)
        state = self.game.reset()
        self._last_obs = state.astype(np.float32)
        # Reset loop detection buffers
        self._action_hist.clear()
        self._in_loop_prev = False
        self._loop_repeats_prev = 0
        self._loop_period_prev = None
        info = {}
        return self._last_obs, info

    def _detect_loop(self):
        """Detect if recent actions form a repeating cycle.
        Returns (in_loop: bool, repeats: int, period: Optional[int]).
        - period P is in [loop_min_period, loop_max_period]
        - repeats is number of completed cycles beyond the first (>=1 means loop)
        """
        n = len(self._action_hist)
        if n < 2 * self.loop_min_period:
            return False, 0, None
        # Use the smallest matching period for stability
        for P in range(self.loop_min_period, self.loop_max_period + 1):
            if n < 2 * P:
                continue
            pattern = list(self._action_hist)[-P:]
            # Check at least two copies exist
            if list(self._action_hist)[-2*P:-P] != pattern:
                continue
            # Count how many consecutive copies at the tail
            repeats = 2  # we already have two
            while n >= (repeats + 1) * P and list(self._action_hist)[-(repeats + 1)*P: -repeats*P] == pattern:
                repeats += 1
            # repeats is number of cycles; we define repeat_count = repeats - 1
            return True, repeats - 1, P
        return False, 0, None

    def step(self, action):
        # Distance-to-food shaping: small bonus for moving closer to current food
        # Use Manhattan distance to the current (pre-step) food target
        old_head = self.game.snake[0]
        old_food = self.game.food
        d_old = None
        if old_food is not None:
            d_old = abs(old_head[0] - old_food[0]) + abs(old_head[1] - old_food[1])

        next_state, reward, done = self.game.step(int(action))
        ate_flag = getattr(self.game, "ate_last_step", False)

        if self.shaping_coef != 0.0:
            # Robust apple detection: use precise flag from game
            ate = ate_flag
            if not ate and d_old is not None and self.game.food is not None:
                new_head = self.game.snake[0]
                d_new = abs(new_head[0] - self.game.food[0]) + abs(new_head[1] - self.game.food[1])
                shaping = self.shaping_coef * (d_old - d_new)
                reward = float(reward + shaping)
        # --- Loop penalty and end-of-loop bonus ---
        loop_pen = 0.0
        loop_bonus = 0.0
        if self.loop_penalty_coef != 0.0 or self.loop_end_bonus != 0.0:
            # Update action history and detect loop at post-action state
            self._action_hist.append(int(action))
            in_loop, repeat_count, period = self._detect_loop()
            # Apply per-step penalty that increases with each additional repeat
            if in_loop and self.loop_penalty_coef != 0.0 and repeat_count > 0:
                loop_pen = - self.loop_penalty_coef * float(repeat_count)
                reward = float(reward + loop_pen)
            # If we just exited a loop this step, give a one-time bonus
            if (not in_loop) and self._in_loop_prev and self.loop_end_bonus != 0.0 and self._loop_repeats_prev > 0:
                loop_bonus = self.loop_end_bonus * float(self._loop_repeats_prev)
                reward = float(reward + loop_bonus)
            # Remember for next step
            self._in_loop_prev = in_loop
            self._loop_repeats_prev = repeat_count if in_loop else 0
            self._loop_period_prev = period if in_loop else None
        self._last_obs = next_state.astype(np.float32)
        info = {"score": self.game.score, "ate": bool(ate_flag)}
        # Add loop diagnostics to info
        if self.loop_penalty_coef != 0.0 or self.loop_end_bonus != 0.0:
            info.update({
                "Loop.is_loop": bool(self._in_loop_prev),
                "Loop.period": int(self._loop_period_prev) if self._loop_period_prev is not None else None,
                "Loop.repeats": int(self._loop_repeats_prev),
                "Loop.penalty": float(loop_pen),
                "Loop.end_bonus": float(loop_bonus),
            })
        terminated = done
        truncated = False
        return self._last_obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "ansi":
            return
        # simple text render
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Score: {self.game.score}")
        self.game._display()

    # --- Curriculum hooks ---
    def apply_params(self, *, width: Optional[int] = None, height: Optional[int] = None, num_apples: Optional[int] = None, step_penalty: Optional[float] = None, apple_reward: Optional[float] = None, death_penalty: Optional[float] = None):
        """Re-create the underlying SnakeGame with new parameters and update observation_space."""
        params = {
            "width": width if width is not None else self.game.width,
            "height": height if height is not None else self.game.height,
            "wall_collision": self.game.wall_collision,
            "num_apples": int(num_apples) if num_apples is not None else getattr(self.game, "num_apples", 1),
            "step_penalty": float(step_penalty) if step_penalty is not None else self.game.step_penalty,
            "apple_reward": float(apple_reward) if apple_reward is not None else self.game.apple_reward,
            "death_penalty": float(death_penalty) if death_penalty is not None else self.game.death_penalty,
        }
        self.game = SnakeGame(**params)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.game.height, self.game.width, 3), dtype=np.float32)


class EpisodeRecorder(gym.Wrapper):
    """Gymnasium wrapper that records full episodes (states, actions, rewards)
    and saves them to disk as compressed .npz files.

    It stores the pre-action state for each step, along with the action and reward,
    so len(states) == len(actions) == len(rewards) == episode length.
    """

    def __init__(self, env: gym.Env, save_dir: str, max_episodes: int = 0, prefix: str = "train"):
        super().__init__(env)
        self.save_dir = save_dir
        self.max_episodes = int(max(0, max_episodes))
        self.prefix = prefix
        self.enabled = self.max_episodes > 0
        self._states = []
        self._actions = []
        self._rewards = []
        self._saved = 0
        self._last_obs = None
        if self.enabled:
            os.makedirs(self.save_dir, exist_ok=True)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        # start fresh episode buffers
        self._states = []
        self._actions = []
        self._rewards = []
        return obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if self.enabled:
            if self._last_obs is not None:
                # record pre-action state
                self._states.append(np.array(self._last_obs))
                self._actions.append(int(action))
                self._rewards.append(float(reward))
        self._last_obs = next_obs

        done = bool(terminated or truncated)
        if self.enabled and done:
            try:
                states = np.stack(self._states, axis=0) if self._states else np.empty((0,), dtype=np.float32)
                actions = np.array(self._actions, dtype=np.int64)
                rewards = np.array(self._rewards, dtype=np.float32)
                meta = {
                    "score": float(info.get("score", 0.0)),
                    "length": int(len(actions)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
                filename = f"{self.prefix}_ep_{self._saved:05d}.npz"
                path = os.path.join(self.save_dir, filename)
                np.savez_compressed(path, states=states, actions=actions, rewards=rewards, meta=meta)
                self._saved += 1
                if self._saved >= self.max_episodes:
                    # disable further recording to avoid overhead
                    self.enabled = False
            except Exception as e:
                # Do not crash training due to logging
                print(f"EpisodeRecorder save error: {e}")

        return next_obs, reward, terminated, truncated, info


class ScoreLoggerCallback(BaseCallback):
    """Logs per-episode score and a running mean to TensorBoard.
    Relies on Monitor wrapper to provide episode_info.
    """

    def __init__(self, mean_window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.recent_scores = deque(maxlen=int(mean_window))

    def _on_step(self) -> bool:
        # When an episode ends, Monitor stores infos in self.locals["infos"] or callback event provides episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # Monitor puts {'r': ep_reward, 'l': length, 't': time}
                # We log separate scoreboard metric from env info if available
                score = info.get("score") or info.get("episode").get("score") if isinstance(info.get("episode"), dict) else None
                # Fallback: try to extract from last info with 'score'
                if score is None and "score" in info:
                    score = info["score"]
                if score is not None:
                    self.recent_scores.append(float(score))
                    mean_score = float(np.mean(self.recent_scores)) if len(self.recent_scores) > 0 else float(score)
                    if self.logger is not None:
                        self.logger.record("rollout/score", float(score))
                        self.logger.record("rollout/score_mean", mean_score)
        return True


class LoopLoggerCallback(BaseCallback):
    """Logs loop detection metrics and start/end events to TensorBoard.

    Expects env infos to include keys populated by SnakeEnv:
      - 'Loop.is_loop': bool
      - 'Loop.period': Optional[int]
      - 'Loop.repeats': int
      - 'Loop.penalty': float (negative when applied)
      - 'Loop.end_bonus': float (positive when awarded)
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.prev_in_loop = False
        self.prev_repeats = 0
        self.episode_loop_starts = 0
        self.episode_loop_ends = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            # Per-step loop metrics
            if "Loop.is_loop" in info:
                in_loop = bool(info.get("Loop.is_loop", False))
                repeats = int(info.get("Loop.repeats", 0) or 0)
                period = info.get("Loop.period", None)
                penalty = float(info.get("Loop.penalty", 0.0) or 0.0)
                end_bonus = float(info.get("Loop.end_bonus", 0.0) or 0.0)

                # Record scalars
                if self.logger is not None:
                    self.logger.record("rollout/loop_is_loop", 1.0 if in_loop else 0.0)
                    self.logger.record("rollout/loop_repeats", float(repeats))
                    if period is not None:
                        self.logger.record("rollout/loop_period", float(period))
                    if penalty != 0.0:
                        self.logger.record("rollout/loop_penalty", penalty)
                    if end_bonus != 0.0:
                        self.logger.record("rollout/loop_end_bonus", end_bonus)

                # Detect start/end events
                if (not self.prev_in_loop) and in_loop:
                    self.episode_loop_starts += 1
                    if self.logger is not None:
                        self.logger.record("rollout/loop_start_event", 1)
                if self.prev_in_loop and (not in_loop):
                    self.episode_loop_ends += 1
                    if self.logger is not None:
                        self.logger.record("rollout/loop_end_event", 1)

                self.prev_in_loop = in_loop
                self.prev_repeats = repeats

            # At episode end, log episode-level counts and reset
            if "episode" in info:
                if self.logger is not None:
                    self.logger.record("rollout/loops_started_ep", float(self.episode_loop_starts))
                    self.logger.record("rollout/loops_ended_ep", float(self.episode_loop_ends))
                self.episode_loop_starts = 0
                self.episode_loop_ends = 0
        return True


class WandbCallback(BaseCallback):
    """Custom callback that logs metrics to Weights & Biases.
    
    Logs training metrics, episode statistics, and loop detection data.
    Works in conjunction with TensorBoard logging.
    
    Note: When sync_tensorboard=True, we don't pass explicit step values
    to avoid conflicts. Instead, we let TensorBoard syncing handle steps.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0
        self.metrics_defined = False

    def _on_training_start(self) -> None:
        """Define custom metrics at the start of training."""
        if not self.metrics_defined and wandb.run is not None:
            # Define custom step metric for our additional metrics
            # This allows W&B to properly track steps even with tensorboard syncing
            wandb.define_metric("episode/*", step_metric="global_step")
            wandb.define_metric("custom/*", step_metric="global_step")
            self.metrics_defined = True

    def _on_step(self) -> bool:
        # When using sync_tensorboard=True, don't pass step parameter to wandb.log
        # The tensorboard integration will handle step tracking automatically
        
        # Log from the logger's current name_to_value dict
        if self.logger is not None:
            log_dict = {}
            # Get all logged values from SB3's logger
            for key in self.logger.name_to_value:
                value = self.logger.name_to_value[key]
                log_dict[key] = value
            
            # Add global step for custom metrics
            if log_dict:
                log_dict["global_step"] = self.num_timesteps
                # Log without explicit step parameter
                wandb.log(log_dict)
        
        # Also log any info dict values
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_count += 1
                # Log episode-level metrics
                ep_info = info["episode"]
                log_dict = {
                    "episode/reward": ep_info.get("r", 0.0),
                    "episode/length": ep_info.get("l", 0),
                    "episode/time": ep_info.get("t", 0.0),
                    "episode/count": self.episode_count,
                    "global_step": self.num_timesteps,
                }
                
                # Log score if available
                if "score" in info:
                    log_dict["episode/score"] = info["score"]
                
                # Log without explicit step parameter
                wandb.log(log_dict)
        
        return True


class CustomEvalCallback(EvalCallback):
    """EvalCallback that also logs loop metrics during evaluation under the 'eval/' namespace.

    Works when the evaluated env exposes loop info in its step info dicts, as provided by SnakeEnv.
    Logs metrics only after evaluation completes, not during training steps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_infos = []

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Called by evaluate_policy during evaluation - collect infos here."""
        infos = locals_.get("infos", [])
        if infos:
            self.eval_infos.extend(infos)

    def update_child_locals(self, locals_: dict) -> None:
        """Called after evaluation to update metrics - log our custom metrics here."""
        super().update_child_locals(locals_)
        
        # Now log all collected eval metrics
        for info in self.eval_infos:
            if isinstance(info, dict):
                # Log loop metrics
                if "Loop.is_loop" in info:
                    in_loop = bool(info.get("Loop.is_loop", False))
                    repeats = int(info.get("Loop.repeats", 0) or 0)
                    period = info.get("Loop.period", None)
                    penalty = float(info.get("Loop.penalty", 0.0) or 0.0)
                    end_bonus = float(info.get("Loop.end_bonus", 0.0) or 0.0)
                    if self.logger is not None:
                        self.logger.record("eval/loop_is_loop", 1.0 if in_loop else 0.0)
                        self.logger.record("eval/loop_repeats", float(repeats))
                        self.logger.record("eval/loop_period", float(period) if period is not None else 0.0)
                        self.logger.record("eval/loop_penalty", penalty)
                        self.logger.record("eval/loop_end_bonus", end_bonus)

                # Log termination and score metrics
                score = info.get("score")
                terminated = info.get("terminated", False)
                truncated = info.get("truncated", False)
                ate = info.get("ate", False)
                if self.logger is not None:
                    if score is not None:
                        self.logger.record("eval/score", float(score))
                    self.logger.record("eval/terminated", 1.0 if terminated else 0.0)
                    self.logger.record("eval/truncated", 1.0 if truncated else 0.0)
                    self.logger.record("eval/ate", 1.0 if ate else 0.0)

                # Log episode-level metrics
                if "episode" in info:
                    ep_info = info["episode"]
                    ep_reward = ep_info.get("r", 0.0)
                    ep_length = ep_info.get("l", 0)
                    ep_time = ep_info.get("t", 0.0)
                    if self.logger is not None:
                        self.logger.record("eval/episode_reward", float(ep_reward))
                        self.logger.record("eval/episode_length", float(ep_length))
                        self.logger.record("eval/episode_time", float(ep_time))

        # Clear for next eval
        self.eval_infos = []


class CurriculumManager:
    """Loads a simple JSON curriculum and returns per-episode parameters.

    JSON schema (minimal):
    {
      "stages": [
        {"from_episode": 0,   "width": 8,  "height": 8,  "num_apples": 1, "max_steps": 200},
        {"from_episode": 100, "width": 12, "height": 12, "num_apples": 2, "max_steps": 300}
      ],
      "target_obs": {"width": 12, "height": 12}  // optional fixed observation size presented to the agent
    }
    """

    def __init__(self, config: dict):
        stages = sorted(config.get("stages", []), key=lambda s: int(s.get("from_episode", 0)))
        assert stages, "Curriculum config must define non-empty 'stages'"
        self.stages = stages
        # Determine fixed observation size for the agent
        if "target_obs" in config:
            self.target_w = int(config["target_obs"].get("width"))
            self.target_h = int(config["target_obs"].get("height"))
        else:
            # default to max over stages
            self.target_w = max(int(s.get("width", DEFAULT_WIDTH)) for s in stages)
            self.target_h = max(int(s.get("height", DEFAULT_HEIGHT)) for s in stages)

    def get_for_episode(self, ep_idx: int) -> dict:
        current = self.stages[0]
        for s in self.stages:
            if ep_idx >= int(s.get("from_episode", 0)):
                current = s
            else:
                break
        return current


class CurriculumWrapper(gym.Wrapper):
    """Wrapper that applies per-episode curriculum updates and pads/crops observations
    to a fixed size so the agent sees a constant observation_space.
    """

    def __init__(self, env: SnakeEnv, manager: CurriculumManager, fallback_max_steps: int):
        super().__init__(env)
        self.manager = manager
        self.episode_idx = -1
        self.current_max_steps = int(fallback_max_steps)
        self.step_count = 0
        H, W = manager.target_h, manager.target_w
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(H, W, 3), dtype=np.float32)

    @staticmethod
    def _pad_or_crop(state: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w, c = state.shape
        # Crop if larger
        state_c = state[:target_h, :target_w, :]
        # Pad if smaller (bottom/right) with empty cells channel
        pad_h = max(0, target_h - state_c.shape[0])
        pad_w = max(0, target_w - state_c.shape[1])
        if pad_h > 0 or pad_w > 0:
            pad = np.zeros((pad_h, state_c.shape[1], 3), dtype=state.dtype)
            if pad_h > 0:
                state_c = np.concatenate([state_c, pad], axis=0)
            if pad_w > 0:
                # recompute because state_c may have changed
                pad2 = np.zeros((state_c.shape[0], pad_w, 3), dtype=state.dtype)
                state_c = np.concatenate([state_c, pad2], axis=1)
            # set padded area to empty channel
            empty_mask = (state_c.sum(axis=2, keepdims=True) == 0)
            state_c[empty_mask.squeeze(-1), 2] = 1.0
        return state_c

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # New episode
        self.episode_idx += 1
        self.step_count = 0
        cfg = self.manager.get_for_episode(self.episode_idx)
        # Apply env params
        width = cfg.get("width")
        height = cfg.get("height")
        num_apples = cfg.get("num_apples")
        step_penalty = cfg.get("step_penalty")
        apple_reward = cfg.get("apple_reward")
        death_penalty = cfg.get("death_penalty")
        self.current_max_steps = int(cfg.get("max_steps", self.current_max_steps))
        # Reconfigure the base env
        self.env.apply_params(width=width, height=height, num_apples=num_apples, step_penalty=step_penalty, apple_reward=apple_reward, death_penalty=death_penalty)
        obs, info = self.env.reset(seed=seed, options=options)
        # Normalize shape to fixed target
        obs_fixed = self._pad_or_crop(obs, self.manager.target_h, self.manager.target_w)
        info = dict(info)
        info["Curriculum.episode"] = self.episode_idx
        return obs_fixed, info

    def step(self, action):
        self.step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Enforce dynamic per-episode max steps
        if not terminated and not truncated and self.step_count >= self.current_max_steps:
            truncated = True
            info = dict(info)
            info["Curriculum.truncated_max_steps"] = True
        obs_fixed = self._pad_or_crop(obs, self.manager.target_h, self.manager.target_w)
        return obs_fixed, reward, terminated, truncated, info


class FixedObsSizeWrapper(gym.ObservationWrapper):
    """Pads/crops observations to (target_h, target_w, 3) and updates observation_space.
    Does not alter environment parameters or step limits. Useful to keep eval obs
    compatible with a curriculum-trained policy.
    """

    def __init__(self, env: gym.Env, target_h: int, target_w: int):
        super().__init__(env)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.target_h, self.target_w, 3), dtype=np.float32)

    @staticmethod
    def _pad_or_crop(state: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w, c = state.shape
        state_c = state[:target_h, :target_w, :]
        pad_h = max(0, target_h - state_c.shape[0])
        pad_w = max(0, target_w - state_c.shape[1])
        if pad_h > 0 or pad_w > 0:
            # pad bottom then right with empty cells
            if pad_h > 0:
                pad = np.zeros((pad_h, state_c.shape[1], 3), dtype=state.dtype)
                state_c = np.concatenate([state_c, pad], axis=0)
            if pad_w > 0:
                pad2 = np.zeros((state_c.shape[0], pad_w, 3), dtype=state.dtype)
                state_c = np.concatenate([state_c, pad2], axis=1)
            empty_mask = (state_c.sum(axis=2, keepdims=True) == 0)
            state_c[empty_mask.squeeze(-1), 2] = 1.0
        return state_c

    def observation(self, obs):
        return self._pad_or_crop(obs, self.target_h, self.target_w)


def make_muon_optimizer_factory(muon_lr: float, aux_lr: float):
    """Create an optimizer factory that uses Muon for hidden weights and Adam for other params.
    
    Args:
        muon_lr: Learning rate for Muon (hidden weights in transformer)
        aux_lr: Learning rate for AdamW (embeddings, output heads, biases/gains)
    
    Returns:
        A function that takes model parameters and returns a Muon optimizer instance.
    """
    def optimizer_factory(params):
        """Factory function compatible with stable-baselines3.
        
        Args:
            params: Iterator of model parameters (from model.parameters())
        
        Returns:
            Muon optimizer instance with appropriate parameter groups.
        """
        # Convert params iterator to list
        all_params = list(params)
        
        # Separate parameters by type:
        # - Hidden weights (2D+ tensors): use Muon
        # - Everything else (embeddings, biases, 1D params): use Adam
        hidden_weights = []
        aux_params = []
        
        for p in all_params:
            if not p.requires_grad:
                continue
            # Muon is designed for 2D+ weight matrices in transformer layers
            if p.ndim >= 2:
                hidden_weights.append(p)
            else:
                aux_params.append(p)
        
        # Build parameter groups for MuonWithAuxAdam
        param_groups = []
        
        if len(hidden_weights) > 0:
            param_groups.append({
                'params': hidden_weights,
                'use_muon': True,
                'lr': muon_lr,
            })
        
        if len(aux_params) > 0:
            param_groups.append({
                'params': aux_params,
                'use_muon': False,
                'lr': aux_lr,
                'betas': (0.9, 0.95),  # Adam betas for aux params
            })
        
        if len(param_groups) == 0:
            raise ValueError("No trainable parameters found!")
        
        print(f"Muon optimizer: {len(hidden_weights)} hidden weight tensors (lr={muon_lr}), "
              f"{len(aux_params)} auxiliary params (lr={aux_lr})")
        
        return MuonWithAuxAdam(param_groups)
    
    return optimizer_factory


def train_sb3(
    width: int,
    height: int,
    num_episodes: int,
    batch_size: int,
    gamma: float,
    d_model: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    lr: float,
    lr_schedule: str,
    lr_end: float,
    log_interval: int,
    max_steps: int,
    model_path: str,
    seed: int,
    eval_episodes: int,
    eval_max_steps: int,
    device: str = "auto",
    # PPO-specific parameters
    n_epochs: int = 10,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    clip_range: float = 0.2,
    gae_lambda: float = 0.95,
    record_dir: Optional[str] = None,
    record_episodes: int = 0,
    eval_record_dir: Optional[str] = None,
    eval_record_episodes: int = 0,
    step_penalty: Optional[float] = None,
    shaping_coef: float = 0.0,
    apple_reward: Optional[float] = None,
    death_penalty: Optional[float] = None,
    max_score: Optional[int] = None,
    curriculum_path: Optional[str] = None,
    # Loop shaping
    loop_penalty_coef: float = 0.0,
    loop_end_bonus: float = 0.0,
    loop_min_period: int = 4,
    loop_max_period: int = 20,
    tensorboard_log: str = "./tb_snake",
    use_wandb: bool = False,
    wandb_project: str = "snake-rl",
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[list] = None,
    # Performance optimization overrides
    no_compile: bool = False,
    no_flash_attention: bool = False,
    no_bf16: bool = False,
    # Muon optimizer parameters
    use_muon: bool = False,
    muon_lr: float = 0.02,
    aux_adam_lr: Optional[float] = None,
):
    set_seed(seed)
    
    run_name = None
    
    # Determine which optimizations to use (env vars can be overridden by CLI flags)
    use_compile = ENV_USE_TORCH_COMPILE and not no_compile
    use_flash = ENV_USE_FLASH_ATTENTION and not no_flash_attention
    use_bf16 = ENV_USE_BF16 and not no_bf16
    
    # Initialize Weights & Biases if requested
    if use_wandb:
        wandb_config = {
            "algorithm": "PPO",
            "width": width,
            "height": height,
            "num_episodes": num_episodes,
            "batch_size": batch_size,
            "gamma": gamma,
            "n_epochs": n_epochs,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "clip_range": clip_range,
            "gae_lambda": gae_lambda,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "lr": lr,
            "lr_schedule": lr_schedule,
            "lr_end": lr_end,
            "max_steps": max_steps,
            "seed": seed,
            "step_penalty": step_penalty,
            "shaping_coef": shaping_coef,
            "apple_reward": apple_reward,
            "death_penalty": death_penalty,
            "max_score": max_score,
            "loop_penalty_coef": loop_penalty_coef,
            "loop_end_bonus": loop_end_bonus,
            "loop_min_period": loop_min_period,
            "loop_max_period": loop_max_period,
            # Performance optimizations
            "use_torch_compile": use_compile,
            "use_flash_attention": use_flash,
            "use_bf16": use_bf16,
        }
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            tags=wandb_tags,
            sync_tensorboard=True,  # Also sync tensorboard logs
        )
        run_name = wandb.run.name if wandb.run else None
    
    # Wrap as Monitor(TimeLimit(...)) so time-limit truncations are recorded consistently
    snake_env = SnakeEnv(
        width=width,
        height=height,
        step_penalty=step_penalty,
        shaping_coef=shaping_coef,
        apple_reward=apple_reward,
        death_penalty=death_penalty,
        num_apples=1,
        render_mode="ansi",
        loop_penalty_coef=loop_penalty_coef,
        loop_end_bonus=loop_end_bonus,
        loop_min_period=loop_min_period,
        loop_max_period=loop_max_period,
    )
    if max_score is not None:
        snake_env = ScoreLimit(snake_env, max_score=max_score)
    # If curriculum provided, wrap with CurriculumWrapper to control max_steps dynamically and normalize obs size
    if curriculum_path:
        import json
        with open(curriculum_path, "r") as f:
            cfg = json.load(f)
        manager = CurriculumManager(cfg)
        base_env = CurriculumWrapper(snake_env, manager=manager, fallback_max_steps=max_steps)
    else:
        base_env = TimeLimit(snake_env, max_episode_steps=max_steps)
    if record_dir and record_episodes > 0:
        base_env = EpisodeRecorder(base_env, save_dir=record_dir, max_episodes=record_episodes, prefix="train")
    env = Monitor(base_env)
    # Seed env deterministically
    env.reset(seed=seed)
    # Separate cap for eval steps to keep eval quick
    _eval_max_steps = eval_max_steps if eval_max_steps and eval_max_steps > 0 else max_steps
    snake_eval_env = SnakeEnv(
        width=width,
        height=height,
        step_penalty=step_penalty,
        shaping_coef=shaping_coef,
        apple_reward=apple_reward,
        death_penalty=death_penalty,
        num_apples=1,
        render_mode="ansi",
        loop_penalty_coef=loop_penalty_coef,
        loop_end_bonus=loop_end_bonus,
        loop_min_period=loop_min_period,
        loop_max_period=loop_max_period,
    )
    if max_score is not None:
        snake_eval_env = ScoreLimit(snake_eval_env, max_score=max_score)
    base_eval_env = TimeLimit(snake_eval_env, max_episode_steps=_eval_max_steps)
    # Match observation size to training env when using curriculum
    if curriculum_path:
        # Reuse the same manager target dims
        base_eval_env = FixedObsSizeWrapper(base_eval_env, target_h=manager.target_h, target_w=manager.target_w)
    if eval_record_dir and eval_record_episodes > 0:
        base_eval_env = EpisodeRecorder(base_eval_env, save_dir=eval_record_dir, max_episodes=eval_record_episodes, prefix="eval")
    eval_env = Monitor(base_eval_env)
    eval_env.reset(seed=seed + 1)

    policy_kwargs = dict(
        features_extractor_class=TransformerExtractor,
        features_extractor_kwargs=dict(d_model=d_model, n_layers=num_layers, n_heads=num_heads, dropout=dropout, features_dim=128),
        net_arch=dict(pi=[128], vf=[128]),  # Separate networks for policy and value function
    )

    # Resolve learning rate (constant vs cosine schedule)
    if (lr_schedule or "constant").lower() == "cosine":
        # Cosine decay from lr -> lr_end over training progress
        learning_rate = cosine_schedule(float(lr), float(lr_end))
    else:
        learning_rate = float(lr)

    # PPO-specific hyperparameters
    n_steps = max_steps  # Number of steps to run for each environment per update
    max_grad_norm = 0.5  # Maximum gradient norm for gradient clipping
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=clip_range,  # Clipping parameter for PPO
        ent_coef=ent_coef,  # Entropy bonus coefficient
        vf_coef=vf_coef,  # Value function loss coefficient
        max_grad_norm=max_grad_norm,
        tensorboard_log=tensorboard_log,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        seed=seed,
    )
    
    # Replace optimizer with Muon if requested
    if use_muon:
        if not _MUON_AVAILABLE:
            print("WARNING: --use-muon set but Muon library not available. Falling back to Adam.")
        else:
            # Initialize torch.distributed for Muon (required by MuonWithAuxAdam)
            import torch.distributed as dist
            if dist.is_available() and not dist.is_initialized():
                try:
                    backend = "nccl" if torch.cuda.is_available() else "gloo"
                    init_file = f"/tmp/muon_pg_{os.getpid()}"
                    dist.init_process_group(backend=backend, init_method=f"file://{init_file}", rank=0, world_size=1)
                    print("Initialized torch.distributed (single-process) for Muon.")
                except Exception as e:
                    print(f"WARNING: Could not initialize torch.distributed for Muon: {e}")
                    print("Falling back to Adam optimizer.")
                    use_muon = False
            
            if use_muon:
                # Determine auxiliary Adam learning rate
                aux_lr = aux_adam_lr if aux_adam_lr is not None else lr
                print(f"Using Muon optimizer: hidden_lr={muon_lr}, aux_lr={aux_lr}")
                
                # Create Muon optimizer and replace the default optimizer
                optimizer_factory = make_muon_optimizer_factory(muon_lr, aux_lr)
                model.policy.optimizer = optimizer_factory(model.policy.parameters())

    # Apply BF16 mixed precision by wrapping policy forward methods
    if use_bf16 and torch.cuda.is_available():
        original_forward = model.policy.forward
        original_evaluate_actions = model.policy.evaluate_actions
        original_predict_values = model.policy.predict_values
        
        def bf16_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                result = original_forward(*args, **kwargs)
            # Convert outputs back to float32 for buffer compatibility
            if isinstance(result, tuple):
                return tuple(x.float() if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16 else x for x in result)
            return result.float() if isinstance(result, torch.Tensor) and result.dtype == torch.bfloat16 else result
        
        def bf16_evaluate_actions(obs, actions):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                values, log_prob, entropy = original_evaluate_actions(obs, actions)
            # Ensure outputs are float32
            return values.float(), log_prob.float(), entropy.float() if entropy is not None else None
        
        def bf16_predict_values(obs):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                values = original_predict_values(obs)
            return values.float()
        
        model.policy.forward = bf16_forward
        model.policy.evaluate_actions = bf16_evaluate_actions
        model.policy.predict_values = bf16_predict_values

    # Apply optimizations from .env configuration (or CLI overrides)
    if use_compile and hasattr(torch, 'compile'):
        print("🔧 Applying torch.compile() optimization...")
        model.policy.features_extractor = torch.compile(
            model.policy.features_extractor,
            mode="reduce-overhead",
        )
        # PPO has separate actor and critic networks
        if hasattr(model.policy, 'mlp_extractor'):
            model.policy.mlp_extractor = torch.compile(model.policy.mlp_extractor, mode="reduce-overhead")
        if hasattr(model.policy, 'action_net'):
            model.policy.action_net = torch.compile(model.policy.action_net, mode="reduce-overhead")
        if hasattr(model.policy, 'value_net'):
            model.policy.value_net = torch.compile(model.policy.value_net, mode="reduce-overhead")
        print("   ✓ Model compiled successfully")

    # Helper function to create SDPA context for FlashAttention
    def get_sdpa_context():
        if use_flash and hasattr(torch.nn.attention, 'sdpa_kernel'):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            print("🔧 Using FlashAttention backend for SDPA")
            return sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION])
        return nullcontext()

    # Print optimization summary
    print("\n" + "="*60)
    print("Training Configuration Summary:")
    print("="*60)
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {'Muon' if (use_muon and _MUON_AVAILABLE) else 'Adam'}")
    if use_muon and _MUON_AVAILABLE:
        print(f"  Muon LR (hidden): {muon_lr}")
        aux_lr_display = aux_adam_lr if aux_adam_lr is not None else lr
        print(f"  Adam LR (aux): {aux_lr_display}")
    print(f"torch.compile: {'✓ Enabled' if use_compile else '✗ Disabled'}")
    print(f"FlashAttention: {'✓ Enabled' if use_flash else '✗ Disabled'}")
    print(f"BF16 (mixed precision): {'✓ Enabled' if use_bf16 else '✗ Disabled'}")
    print("="*60 + "\n")

    # Use SB3's built-in EvalCallback for periodic evaluation and best-model saving
    eval_cb = CustomEvalCallback(
        eval_env,
        n_eval_episodes=eval_episodes,
        eval_freq=log_interval * max_steps,
        best_model_save_path="./models",
        deterministic=True,
        render=False,
    )

    # Add SB3's built-in progress bar (shows total timesteps progress)
    pbar_cb = ProgressBarCallback()

    total_timesteps = num_episodes * max_steps
    try:
        score_cb = ScoreLoggerCallback(mean_window=100)
        loop_cb = LoopLoggerCallback()
        callbacks = [pbar_cb, eval_cb, score_cb, loop_cb]
        
        # Add WandbCallback if using wandb
        if use_wandb:
            wandb_cb = WandbCallback()
            callbacks.append(wandb_cb)
        
        # Apply SDPA optimizations during training
        # Note: BF16 is already applied to policy methods if enabled
        with get_sdpa_context():
            if use_bf16 and torch.cuda.is_available():
                print("🚀 Starting training with BF16 mixed precision...")
            else:
                print("🚀 Starting training with FP32 precision...")
            model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    finally:
        # Clean up torch.distributed if initialized for Muon
        if use_muon and _MUON_AVAILABLE:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
        
        # Ensure the terminal cursor is visible again
        print("\033[?25h", end="", flush=True)
        
        # Handle model path formatting and saving
        if run_name is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(model_path)
            model_path = f"{base}_{run_name}_{timestamp}{ext}"
            if not model_path.endswith(".zip"):
                model_path = f"{model_path}.zip"
            # Log to wandb if available
            if use_wandb:
                wandb.log({"final_model_path": model_path})
        
        # Save the model
        model_dir = os.path.dirname(model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        model.save(model_path)
        
        # Finish wandb run if initialized
        if use_wandb:
            wandb.finish()

    # Save some logs like original
    # SB3 logs to tensorboard, but we can save episode logs if needed
    # For simplicity, skip custom logs for now

def play_sb3(
    model_path: str,
    width: int,
    height: int,
    render_delay: float,
    step_penalty: Optional[float] = None,
    shaping_coef: float = 0.0,
    apple_reward: Optional[float] = None,
    death_penalty: Optional[float] = None,
    max_score: Optional[int] = None,
    # Loop shaping
    loop_penalty_coef: float = 0.0,
    loop_end_bonus: float = 0.0,
    loop_min_period: int = 4,
    loop_max_period: int = 20,
):
    env = SnakeEnv(
        width=width,
        height=height,
        wall_collision=True,
        step_penalty=step_penalty,
        shaping_coef=shaping_coef,
        apple_reward=apple_reward,
        death_penalty=death_penalty,
        render_mode="ansi",
        loop_penalty_coef=loop_penalty_coef,
        loop_end_bonus=loop_end_bonus,
        loop_min_period=loop_min_period,
        loop_max_period=loop_max_period,
    )
    if max_score is not None:
        env = ScoreLimit(env, max_score=max_score)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        time.sleep(render_delay)

def reconstruct_episode_sb3(episode_log, render_delay: float = 0.1):
    # This is unchanged, as it doesn't depend on the model
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
    p = argparse.ArgumentParser(description="Snake with SB3 Transformer PPO")
    sub = p.add_subparsers(dest="mode", required=False)

    # Train args
    pt = sub.add_parser("train", help="Train the agent with SB3 PPO")
    pt.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Grid width (default: {DEFAULT_WIDTH})")
    pt.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Grid height (default: {DEFAULT_HEIGHT})")
    pt.add_argument("--episodes", type=int, default=DEFAULT_NUM_EPISODES, help=f"Number of training episodes (default: {DEFAULT_NUM_EPISODES})")
    pt.add_argument("--batch-size", type=int, default=ENV_BATCH_SIZE, help=f"Minibatch size for PPO updates (default: {ENV_BATCH_SIZE})")
    pt.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help=f"Discount factor (default: {DEFAULT_GAMMA})")
    pt.add_argument("--n-epochs", type=int, default=10, help="Number of epochs for PPO optimization (default: 10)")
    pt.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient for exploration (default: 0.01)")
    pt.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient (default: 0.5)")
    pt.add_argument("--clip-range", type=float, default=0.2, help="PPO clipping parameter (default: 0.2)")
    pt.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter (default: 0.95)")
    pt.add_argument("--d-model", type=int, default=DEFAULT_HIDDEN_DIM, help=f"Transformer model dimension (default: {DEFAULT_HIDDEN_DIM})")
    pt.add_argument("--layers", type=int, default=DEFAULT_NUM_LAYERS, help=f"Number of Transformer layers (default: {DEFAULT_NUM_LAYERS})")
    pt.add_argument("--heads", type=int, default=DEFAULT_NUM_HEADS, help=f"Number of Transformer attention heads (default: {DEFAULT_NUM_HEADS})")
    pt.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help=f"Transformer dropout rate (default: {DEFAULT_DROPOUT})")
    pt.add_argument("--lr", type=float, default=DEFAULT_LR, help=f"Initial learning rate (use with --lr-schedule, default: {DEFAULT_LR})")
    pt.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant", help="Learning rate schedule: constant keeps LR fixed; cosine decays from --lr to --lr-end")
    pt.add_argument("--lr-end", type=float, default=0.0, help="Final learning rate when using --lr-schedule cosine (default: 0.0)")
    pt.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE, help=f"Replay buffer size (default: {DEFAULT_REPLAY_SIZE})")
    pt.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, help=f"Episodes between logging and eval (default: {DEFAULT_LOG_INTERVAL})")
    pt.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"Maximum steps per episode (default: {DEFAULT_MAX_STEPS})")
    pt.add_argument("--model-path", type=str, default="sb3_snake_ppo_transformer.zip")
    pt.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    pt.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device to use for training (default: auto)")
    pt.add_argument("--curriculum", type=str, default=None, help="Path to JSON curriculum file for per-episode schedule")

    # Recording options
    pt.add_argument("--record-dir", type=str, default="eval_logs", help="Directory to save recorded training episodes (.npz)")
    pt.add_argument("--record-episodes", type=int, default=0, help="Number of training episodes to record (0 to disable)")
    pt.add_argument("--eval-record-dir", type=str, default="eval_logs", help="Directory to save recorded eval episodes (.npz)")
    pt.add_argument("--eval-record-episodes", type=int, default=0, help="Number of eval episodes to record (0 to disable)")

    # Eval controls
    pt.add_argument("--eval-episodes", "--eval_episodes", dest="eval_episodes", type=int, default=50, help="Number of episodes per evaluation pass")
    pt.add_argument("--eval-max-steps", "--eval_max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Per-episode max steps during evaluation (caps episode length)")
    pt.add_argument("--check-env", action="store_true", help="Run SB3 env checker before training")

    # Reward shaping controls
    pt.add_argument("--step-penalty", type=float, default=None, help="Per-step penalty to discourage inefficiency (e.g., -0.001 or 0.0). None uses env default (-0.01).")
    pt.add_argument("--shaping-coef", type=float, default=0.0, help="Distance-to-food shaping coefficient (e.g., 0.02)")
    pt.add_argument("--apple-reward", type=float, default=None, help="Reward for eating food (default 10.0)")
    pt.add_argument("--death-penalty", type=float, default=None, help="Penalty for dying (default -10.0)")
    pt.add_argument("--max-score", type=int, default=None, help="Maximum score to cap episode length (None for no limit)")

    # Loop shaping controls
    pt.add_argument("--loop-penalty-coef", type=float, default=0.0, help="Per-step penalty scale when looping; multiplied by repeat count (m-1)")
    pt.add_argument("--loop-end-bonus", type=float, default=0.0, help="One-time bonus awarded when exiting a detected loop; multiplied by last repeat count")
    pt.add_argument("--loop-min-period", type=int, default=4, help="Minimum action cycle length to consider a loop")
    pt.add_argument("--loop-max-period", type=int, default=20, help="Maximum action cycle length to consider a loop")
    
    # Weights & Biases logging
    pt.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    pt.add_argument("--wandb-project", type=str, default="snake-rl", help="W&B project name")
    pt.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name (auto-generated if not specified)")
    pt.add_argument("--wandb-tags", nargs="*", default=None, help="W&B tags for the run")

    # Performance optimization flags (override .env settings)
    pt.add_argument("--no-compile", action="store_true", help="Disable torch.compile() even if enabled in .env")
    pt.add_argument("--no-flash-attention", action="store_true", help="Disable FlashAttention even if enabled in .env")
    pt.add_argument("--no-bf16", action="store_true", help="Disable BF16 mixed precision even if enabled in .env")
    
    # Muon optimizer
    pt.add_argument("--use-muon", action="store_true", default=False, help="Use Muon optimizer for hidden weights (transformer layers).")
    pt.add_argument("--muon-lr", type=float, default=0.02, help="Learning rate for Muon (hidden weights). Default: 0.02")
    pt.add_argument("--aux-adam-lr", type=float, default=None, help="Learning rate for auxiliary AdamW groups (embeddings, heads, gains/biases). Defaults to --lr if not specified.")

    # Play args
    pp = sub.add_parser("play", help="Play using a trained SB3 PPO model")
    pp.add_argument("--model-path", type=str, default="sb3_snake_ppo_transformer.zip")
    pp.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Grid width (default: {DEFAULT_WIDTH})")
    pp.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Grid height (default: {DEFAULT_HEIGHT})")
    pp.add_argument("--render-delay", type=float, default=DEFAULT_RENDER_DELAY, help=f"Delay between steps when rendering (default: {DEFAULT_RENDER_DELAY})")
    # Reward shaping controls during play (for diagnostics/reward display only)
    pp.add_argument("--step-penalty", type=float, default=None, help="Per-step penalty during play (for reward display consistency)")
    pp.add_argument("--shaping-coef", type=float, default=0.0, help="Shaping coefficient during play (has no effect on model outputs, only reward display)")
    pp.add_argument("--apple-reward", type=float, default=None, help="Reward for eating food during play (display only)")
    pp.add_argument("--death-penalty", type=float, default=None, help="Penalty for dying during play (display only)")
    pp.add_argument("--max-score", type=int, default=None, help="Maximum score to cap episode during play (None for no limit)")
    # Loop shaping controls during play (for diagnostics/reward display only)
    pp.add_argument("--loop-penalty-coef", type=float, default=0.0)
    pp.add_argument("--loop-end-bonus", type=float, default=0.0)
    pp.add_argument("--loop-min-period", type=int, default=4)
    pp.add_argument("--loop-max-period", type=int, default=20)

    # Utility: reconstruct episode from logs
    pr = sub.add_parser("reconstruct", help="Playback a recorded episode (.npz file or a directory of .npz)")
    pr.add_argument("--path", type=str, default="eval_logs", help=".npz file path or directory containing .npz episodes")
    pr.add_argument("--index", type=int, default=0, help="When path is a directory, which episode index to load")
    pr.add_argument("--render-delay", type=float, default=DEFAULT_RENDER_DELAY, help=f"Delay between steps when rendering (default: {DEFAULT_RENDER_DELAY})")

    return p

def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "play":
        play_sb3(
            model_path=args.model_path,
            width=args.width,
            height=args.height,
            render_delay=args.render_delay,
            step_penalty=getattr(args, "step_penalty", None),
            shaping_coef=getattr(args, "shaping_coef", 0.0),
            apple_reward=getattr(args, "apple_reward", None),
            death_penalty=getattr(args, "death_penalty", None),
            max_score=getattr(args, "max_score", None),
            loop_penalty_coef=getattr(args, "loop_penalty_coef", 0.0),
            loop_end_bonus=getattr(args, "loop_end_bonus", 0.0),
            loop_min_period=getattr(args, "loop_min_period", 4),
            loop_max_period=getattr(args, "loop_max_period", 20),
        )
        return
    elif args.mode == "reconstruct":
        path = args.path
        if os.path.isdir(path):
            files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.npz')])
            if not files:
                print(f"No .npz episodes found in directory: {path}")
                return
            idx = max(0, min(args.index, len(files) - 1))
            ep = np.load(files[idx], allow_pickle=True)
            reconstruct_episode_sb3(ep, render_delay=args.render_delay)
        elif os.path.isfile(path) and path.endswith('.npz'):
            ep = np.load(path, allow_pickle=True)
            reconstruct_episode_sb3(ep, render_delay=args.render_delay)
        else:
            print(f"Path is neither a directory with .npz files nor a .npz file: {path}")
        return

    # Optional: run env checker before training
    if getattr(args, "mode", None) in (None, "train") and getattr(args, "check_env", False):
        check_env(SnakeEnv(width=args.width, height=args.height, render_mode="ansi"))

    # Default to train
    if not hasattr(args, "width"):
        args = argparse.Namespace(
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            episodes=DEFAULT_NUM_EPISODES,
            batch_size=DEFAULT_BATCH_SIZE,
            gamma=DEFAULT_GAMMA,
            n_epochs=10,
            ent_coef=0.01,
            vf_coef=0.5,
            clip_range=0.2,
            gae_lambda=0.95,
            d_model=DEFAULT_HIDDEN_DIM,
            layers=DEFAULT_NUM_LAYERS,
            heads=DEFAULT_NUM_HEADS,
            dropout=DEFAULT_DROPOUT,
            lr=DEFAULT_LR,
            lr_schedule="constant",
            lr_end=0.0,
            log_interval=DEFAULT_LOG_INTERVAL,
            max_steps=DEFAULT_MAX_STEPS,
            model_path="sb3_snake_ppo_transformer.zip",
            seed=DEFAULT_SEED,
            eval_episodes=5,
            eval_max_steps=DEFAULT_MAX_STEPS,
            step_penalty=None,
            shaping_coef=0.0,
            apple_reward=None,
            death_penalty=None,
            max_score=None,
        )

    train_sb3(
        width=args.width,
        height=args.height,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        d_model=args.d_model,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        lr_schedule=getattr(args, "lr_schedule", "constant"),
        lr_end=getattr(args, "lr_end", 0.0),
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        model_path=args.model_path,
        seed=args.seed,
        device=getattr(args, "device", "auto"),
        eval_episodes=getattr(args, "eval_episodes", 5),
        eval_max_steps=getattr(args, "eval_max_steps", DEFAULT_MAX_STEPS),
        # PPO-specific parameters
        n_epochs=getattr(args, "n_epochs", 10),
        ent_coef=getattr(args, "ent_coef", 0.01),
        vf_coef=getattr(args, "vf_coef", 0.5),
        clip_range=getattr(args, "clip_range", 0.2),
        gae_lambda=getattr(args, "gae_lambda", 0.95),
        record_dir=getattr(args, "record_dir", None),
        record_episodes=getattr(args, "record_episodes", 0),
        eval_record_dir=getattr(args, "eval_record_dir", None),
        eval_record_episodes=getattr(args, "eval_record_episodes", 0),
        step_penalty=getattr(args, "step_penalty", None),
        shaping_coef=getattr(args, "shaping_coef", 0.0),
        apple_reward=getattr(args, "apple_reward", None),
        death_penalty=getattr(args, "death_penalty", None),
        max_score=getattr(args, "max_score", None),
        curriculum_path=getattr(args, "curriculum", None),
        loop_penalty_coef=getattr(args, "loop_penalty_coef", 0.0),
        loop_end_bonus=getattr(args, "loop_end_bonus", 0.0),
        loop_min_period=getattr(args, "loop_min_period", 4),
        loop_max_period=getattr(args, "loop_max_period", 20),
        use_wandb=getattr(args, "wandb", False),
        wandb_project=getattr(args, "wandb_project", "snake-rl"),
        wandb_run_name=getattr(args, "wandb_run_name", None),
        wandb_tags=getattr(args, "wandb_tags", None),
        no_compile=getattr(args, "no_compile", False),
        no_flash_attention=getattr(args, "no_flash_attention", False),
        no_bf16=getattr(args, "no_bf16", False),
        use_muon=getattr(args, "use_muon", False),
        muon_lr=getattr(args, "muon_lr", 0.02),
        aux_adam_lr=getattr(args, "aux_adam_lr", None),
    )

class ScoreLimit(gym.Wrapper):
    """Gymnasium wrapper that truncates episodes when the score reaches a maximum value."""

    def __init__(self, env: gym.Env, max_score: int):
        super().__init__(env)
        self.max_score = max_score

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not terminated and not truncated and info.get("score", 0) >= self.max_score:
            truncated = True
            info["ScoreLimit.truncated"] = True
        return obs, reward, terminated, truncated, info

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting...")
        sys.exit(0)
