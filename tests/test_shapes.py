import numpy as np
import torch

from snake import TransformerModel, DQN


def test_transformer_forward_shapes():
    torch.manual_seed(0)
    B, H, W, C = 2, 10, 10, 3
    S = H * W
    x = torch.randn(B, S, C)

    model = TransformerModel(in_channels=C, action_dim=4, d_model=32, num_layers=1, num_heads=4, dropout=0.0, height=H, width=W)
    out = model(x)
    assert out.shape == (B, 4)


def test_dqn_tokenize_and_forward():
    torch.manual_seed(0)
    np.random.seed(0)
    H, W = 8, 8
    # Build a plausible state: one snake cell, one food, rest empty
    state = np.zeros((H, W, 3), dtype=np.float32)
    state[H//2, W//2, 0] = 1.0  # snake
    state[0, 0, 1] = 1.0        # food
    state[state[...,0] == 0][..., 2] = 1.0  # empty elsewhere

    dqn = DQN(action_dim=4, d_model=32, num_layers=1, num_heads=4, dropout=0.0, lr=1e-3, replay_size=1024, height=H, width=W)
    tokens = dqn.state_to_tokens(state)
    assert tokens.shape == (H*W, 3)
    dev = next(dqn.policy_net.parameters()).device
    logits = dqn.policy_net(tokens.unsqueeze(0).to(dev))  # (1, S, C) -> (1, A)
    assert logits.shape == (1, 4)
