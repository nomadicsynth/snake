# Development Journal - Snake AI

## Implemented Ideas

### [X] Pure RL on Transformer Model with Stable Baselines 3

This was the first iteration of the model, using a transformer architecture to process the grid state and output actions. The model has a DQN head for discrete action selection. Training was done using Stable Baselines 3. That was slow as hell. Then I implemented a `purejaxrl` version, which is about 1000x faster. Then I tried PPO instead of DQN. The model never beat cold-start within the time allocated by my patience. Pure RL with a transformer is not for the impatient.

### [✓] Supervised Pretraining on Expert Dataset

This is the current approach, and it works well. The model learns to imitate expert behavior from a dataset of game states and optimal actions. The dataset is generated using A* pathfinding and safety heuristics. The model architecture is a simple transformer encoder with a classification head for action prediction. Training is done using teacher-forcing and cross-entropy loss. This approach is much faster than pure RL, and the model learns to play well within about 20 epochs on a dataset of 50,000 samples (augmented to 400,000). The pretrained model can then be fine-tuned with RL if desired, although this has not yet been successfully implemented. The RL fine-tuning invariably leads to catastrophic forgetting of the pretrained knowledge. Sweeps never found successful hyperparameters before I got bored. Again, not for the impatient.

### [✓] Transformer with CNN Backbone

A hybrid model that combines convolutional neural networks (CNNs) for initial feature extraction from the grid state, followed by transformer layers for sequence modeling. This approach leverages the strengths of CNNs in capturing spatial hierarchies and transformers in modeling long-range dependencies. Options to replace the transformer input with the CNN output, or to concatenate both representations.

Brief testing showed improvement over the pure transformer model at d_model=64, and that concatenating both representations yielded better results, but haven't tested extensively. Decided to go with the concatenation approach for now, as it seems to work well.

### [WIP] Reasoning-augmented Snake Model (RSM)

An enhanced transformer model that incorporates explicit reasoning about the game state to improve decision-making. Inspired by techniques in reinforcement learning and natural language processing, this model aims to better understand the consequences of actions through structured reasoning.

## Ideas for Improving the Model

### [ ] RL Fine-tuning of Pretrained Model

Fine-tune the pretrained model using reinforcement learning to further improve performance. Traditional RL techniques weren't working, but many have been adapted to LLMs recently, so inspiration can be drawn from examples like the `trl` library. The challenge is to avoid catastrophic forgetting of the pretrained knowledge while still allowing the model to adapt and improve through RL. I particulary want to try `GRPOTrainer` from `trl`. Group-relative policy optimization is basically online PPO-RL for LMs. It can be done with only reward functions, no reward models needed. I think this is called RLVR (Reinforcement learning with verifiable rewards). The reward functions are already in the environment, so it should be straightforward to implement. My experience with GRPO in LMs is that it is very efficient in adapting a model - on the order of thousands of samples, not millions. This is because the model is already pretrained and only needs to make small adjustments. This can be done with the pretraining dataset, or with real game play, depending on what you prefer. Real gameplay is great for complex games, but snake is pretty tractable compared to some others, so it's easy to generate a lot of training data quickly. I guess it just depends on whether the training data generator has the same distribution as real gameplay, which it probably doesn't. Experiment with both and see what works best if patience and hyperfocus sustain.

### [ ] Differential Transformer (Diff-Transformer)

From "Differential Transformer"
Tianzhu Ye, Li Dong, Yuqing Xia, Yutao Sun, Yi Zhu, Gao Huang, Furu Wei
<https://arxiv.org/abs/2410.05258v1>

```text
Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models. 
```

### [ ] Normalised Transformer (nGPT)

Applies L2 norm to the hidden states after every operation. Intuitively, the tokens "travel along a hypersphere". I can't remember the justification, but it sounds poetic so i'll give it a go.

### [ ] EqM (Equilibrium Matching)

From "Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models"
Runqian Wang, Yilun Du
<https://arxiv.org/abs/2510.02300>

```text
We introduce Equilibrium Matching (EqM), a generative modeling framework built from an equilibrium dynamics perspective. EqM discards the non-equilibrium, time-conditional dynamics in traditional diffusion and flow-based generative models and instead learns the equilibrium gradient of an implicit energy landscape. Through this approach, we can adopt an optimization-based sampling process at inference time, where samples are obtained by gradient descent on the learned landscape with adjustable step sizes, adaptive optimizers, and adaptive compute. EqM surpasses the generation performance of diffusion/flow models empirically, achieving an FID of 1.90 on ImageNet 256×256. EqM is also theoretically justified to learn and sample from the data manifold. Beyond generation, EqM is a flexible framework that naturally handles tasks including partially noised image denoising, OOD detection, and image composition. By replacing time-conditional velocities with a unified equilibrium landscape, EqM offers a tighter bridge between flow and energy-based models and a simple route to optimization-driven inference.
```

### [ ] World Models

Experiment with generative modeling of game states, to learn a "world model" that can be used as the core of the game-playing model. This could be a VAE, diffusion model, or other generative architecture. The idea the same as unsupervised language modeling, extended to game-playing. LLMs are effectively world models of text, so this is a natural extension. The weights of the world model can be frozen, and a small policy head can be trained on top, similar to pretraining in NLP. <- or something like that

### [ ] Multi-Agent Snake

Extend the Snake environment to support multiple snakes (agents) on the same grid. Each agent can have its own policy, and they can compete or cooperate for food. This would introduce new dynamics and challenges, such as avoiding collisions with other snakes, strategizing for food, and possibly forming alliances. The model architecture may need to be adapted to handle multiple agents, possibly using multi-head attention or separate policy networks for each agent.

### [ ] Learn to Play Other Games

Extend the current architecture and training framework to other grid-based games. This could potentially be used for ARC AGI 3, which is about playing games, or just general game-playing. The idea is to create a generalist model that can learn to play multiple games, similar to how LLMs can handle multiple tasks. This would involve creating datasets for other games, adapting the model architecture if necessary, and training the model on a diverse set of games.

Eventually I'd like to try actual video games, like Half-Life or Portal. I'm aware of Doom-playing models, and there may be enough power in a single GPU to run a small model. This model runs a 20x20 grid at 15000fps on a 4090, so maybe a 640x480 game at 30fps is feasible. The current model takes RGB frames as input, so it's just a matter of wiring it up in a way it can get frame-grabs and send keypresses. If it can run fast enough it might even be able to reason in an outer-loop or second reasoning model while another module actually controls the game and execites the reasoning modules plans.

### [ ] Real-time Strategy (RTS) Games

Extend the model to handle real-time strategy games, which involve managing resources, building units, and strategizing against opponents in real-time. This would require adapting the model to handle continuous time and possibly a larger action space. The model could be trained on datasets of RTS game states and actions, or through self-play. This would be a significant challenge, but could lead to interesting insights into multi-agent coordination and long-term planning.
