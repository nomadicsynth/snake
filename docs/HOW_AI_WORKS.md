# How AI Works - A Video Explanation for my mates

## Background

I'm making a video to explain how AI works, aimed at my mates who aren't familiar with the technical details. I'm thinking of using Snake as an example, since it's simple and visual. The goal is to give them an intuitive understanding of the concepts without overwhelming them with jargon.
gonna do somehting like show the math, a simple version. i explained to my mate this afternoon how you train the model by iteratively adjusting the parameters of a simple but large mathematical formula to make it give you the answer you want. but i forgot to tell him how you train it to predict based on a prompt. all i told him was the most simple version - if i have a variable `a`, and i set it to one, and i want it to be ten, i can just set it to ten. but for a model you have a lot of these variables, so you don't know what to set them to, so you just set them randomly, see what the output is, then make a small adjustment to each one. eventually, after doing that *a lot*, you have a function that does what you want. but i forgot to tell him how you train it to predict based on a prompt. so i need to do that.

## Series Outline

1. how does a neural network work - simple mathematical function with a lot of parameters, training by adjusting parameters to minimize error
2. Attention is all you need - transformers, self-attention, how it works, how it enabled LLMs
3. Games are all you need - why gameplay is the future of AI, probably; games are models of the world, so learning to play games is like learning to understand the world. Snake is just the start; playing games like Portal and The Talos Principle is like learning to think and reason. eventually we get to real-world tasks, but games are a good stepping stone. Step 1 is playing like a person plays, with vision and controls. Step 2 is embodying the model in an avatar in the game world so it gets rich sensory input and can interact with the world in a more natural way.
4. Playing Snake - applying transformers to Snake, how the model sees the grid, how it predicts actions
5. Reasoning-augmented models - how adding reasoning tokens helps the model think more deeply about its actions, inspired by chain-of-thought prompting in LLMs
6. Future directions - fine-tuning with RL, world models, multi-agent systems, learning to play more complex games
