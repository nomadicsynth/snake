# Technical Report Preparation Checklist

**Title**: Equilibrium World Models for Sequential Decision Making  
**Report ID**: NC-TR-2025-001  
**Target Date**: TBD

---

## 📋 Overview

This document outlines what needs to be completed before writing the technical report on EqM-based world models for Snake RL.

## ✅ Prerequisites (What's Already Done)

- [x] EqM world model implementation (`model/model_eqm.py`)
- [x] Training infrastructure (`train_snake_world.py`)
- [x] Dataset generation script (`generate_world_model_dataset.py`)
- [x] Documentation (EQM_WORLD_MODEL.md, EQM_QUICKSTART.md)
- [x] Baseline transformer policy (for comparison)

## 🔬 Experiments to Run

### 1. Baseline Training (Standard Policy)

**Purpose**: Establish comparison baseline

- [ ] Train standard TransformerPolicy on Snake
  - Small scale: 10k samples, 20 epochs
  - Medium scale: 50k samples, 50 epochs
  - Large scale: 200k samples, 100 epochs
- [ ] Record metrics:
  - Training accuracy
  - Validation accuracy
  - Training time
  - Inference speed (actions/sec)
  - Model size (parameters)
- [ ] Save best checkpoint for comparison

**Command**:

```bash
python train_hf.py \
  --dataset outputs/datasets/snake_dataset_baseline/ \
  --epochs 50 \
  --batch-size 256 \
  --output-dir outputs/models/baseline_policy \
  --wandb --wandb-project snake-technical-report \
  --run-name baseline-transformer
```

### 2. EqM World Model Training

**Purpose**: Main experimental results

#### 2.1 Initial Training Run

- [ ] Generate world model dataset (10k samples for quick test)

  ```bash
  python generate_world_model_dataset.py \
    --num-samples 10000 \
    --output snake_world_pilot \
    --use-astar --augment
  ```

- [ ] Train EqM model with default hyperparameters

  ```bash
  python train_snake_world.py \
    --dataset outputs/datasets/snake_world_pilot/ \
    --epochs 20 \
    --batch-size 256 \
    --output-dir outputs/models/eqm_pilot \
    --wandb --wandb-project snake-technical-report \
    --run-name eqm-pilot
  ```

- [ ] Record metrics:
  - EqM loss components (loss_latent, loss_action)
  - Action prediction accuracy
  - Training time
  - Sampling speed (with different step counts)

#### 2.2 Gradient Schedule Ablation

**Purpose**: Compare different c(γ) schedules

- [ ] Linear schedule (baseline)
- [ ] Truncated schedule (a=0.3)
- [ ] Piecewise schedule (a=0.3, b=2.0)

For each:

```bash
python train_snake_world.py \
  --dataset outputs/datasets/snake_world_pilot/ \
  --gradient-schedule [linear|truncated|piecewise] \
  --epochs 20 \
  --wandb --run-name eqm-schedule-{name}
```

#### 2.3 Sampling Steps Ablation

**Purpose**: Trade-off between quality and speed

- [ ] 5 steps
- [ ] 10 steps (default)
- [ ] 20 steps
- [ ] 50 steps
- [ ] Adaptive (threshold=0.01)

Test at evaluation time with best trained model.

#### 2.4 Latent Dimension Ablation

**Purpose**: Find optimal latent space size

- [ ] 32 dims
- [ ] 64 dims (default)
- [ ] 128 dims
- [ ] 256 dims

#### 2.5 Architecture Variants

**Purpose**: Explore design space

- [ ] Without CNN (`--no-cnn`)
- [ ] CNN replace mode (`--cnn-mode replace`)
- [ ] CNN append mode (`--cnn-mode append`, default)
- [ ] Different encoder depths (2, 3, 4 layers)
- [ ] Different EqM depths (2, 3, 4 layers)

### 3. Evaluation Metrics

For each trained model, compute:

#### 3.1 Standard Metrics

- [ ] Action prediction accuracy
- [ ] Training loss curves
- [ ] Validation loss curves
- [ ] Training time (total and per epoch)
- [ ] Model size (parameters, disk space)

#### 3.2 World Model Specific

- [ ] Next-state prediction accuracy
  - Visual similarity (MSE, SSIM)
  - Semantic accuracy (food/snake positions correct?)
- [ ] Gradient norm distribution during sampling
- [ ] Number of sampling steps taken (if adaptive)
- [ ] Latent space visualization (t-SNE of learned states)

#### 3.3 In-Game Performance

- [ ] Play games with trained policy
  - Average score
  - Average game length
  - Success rate (food collected)
  - Death causes (wall, self-collision, timeout)
- [ ] Compare: Baseline policy vs EqM world model
- [ ] Visualize decision making (energy landscape?)

**Command** (once you have play script):

```bash
python play_pretrained.py \
  --model outputs/models/eqm_best/final_model \
  --num-episodes 100 \
  --record-video \
  --output-dir outputs/eval/eqm_gameplay
```

### 4. Qualitative Analysis

- [ ] Visualize the energy landscape
  - Sample multiple next-states from different starting points
  - Show gradient field arrows
  - Highlight minima (predicted states)

- [ ] Optimization trajectory visualization
  - Show how (state, action) evolves during GD
  - Compare different sampling steps
  - Show cases where adaptive stopping helps

- [ ] Failure mode analysis
  - When does the model predict poorly?
  - What do bad minima look like?
  - Edge cases (snake about to die, etc.)

- [ ] Attention visualization
  - What does the encoder attend to?
  - Does it focus on head, food, obstacles?

## 📊 Data to Collect

Create a results spreadsheet/table with:

| Model | Config | Train Acc | Val Acc | Train Time | Params | Game Score | Notes |
|-------|--------|-----------|---------|------------|--------|------------|-------|
| Baseline | standard | ... | ... | ... | ... | ... | ... |
| EqM | linear | ... | ... | ... | ... | ... | ... |
| EqM | truncated | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

## 🎨 Figures to Create

### Required Figures

1. **Architecture Diagram**
   - [ ] High-level flow: Grid → Encoder → EqM → Sample
   - [ ] Detailed: Show transformer, latent projection, gradient field
   - [ ] Training vs inference modes

2. **Training Curves**
   - [ ] Loss over time (train + val)
   - [ ] Accuracy over time
   - [ ] Compare baseline vs EqM

3. **Gradient Schedule Comparison**
   - [ ] Plot c(γ) for linear/truncated/piecewise
   - [ ] Show how it affects training

4. **Sampling Process**
   - [ ] Trajectory of (state, action) during optimization
   - [ ] Gradient norms over steps
   - [ ] Final convergence

5. **Results Comparison**
   - [ ] Bar chart: accuracy across models
   - [ ] Bar chart: game performance
   - [ ] Speed vs quality trade-off (sampling steps)

6. **Qualitative Examples**
   - [ ] Side-by-side: real next_state vs predicted
   - [ ] Energy landscape visualization
   - [ ] Success cases and failure cases

### Optional Figures

1. **Latent Space Visualization**
   - [ ] t-SNE of learned state representations
   - [ ] Color by: snake length, proximity to food, danger

2. **Ablation Study Results**
   - [ ] Heatmap of hyperparameter effects
   - [ ] Sensitivity analysis

3. **Attention Maps**
   - [ ] Show what encoder attends to

## 📝 Writing Components

### Section 1: Introduction

- [ ] Motivation: Why world models matter
- [ ] Problem: Traditional approaches vs EqM
- [ ] Contribution: Joint state-action optimization
- [ ] Results preview: Key findings

### Section 2: Background

- [ ] Energy-Based Models basics
- [ ] Equilibrium Matching (cite original paper)
- [ ] World models in RL
- [ ] Snake as a testbed

### Section 3: Method

- [ ] Architecture details
- [ ] EqM training objective (math)
- [ ] Gradient schedules c(γ)
- [ ] Sampling procedure (Algorithm pseudocode)
- [ ] Implementation details

### Section 4: Experiments

- [ ] Dataset description
- [ ] Training setup
- [ ] Hyperparameters
- [ ] Baseline comparisons
- [ ] Ablation studies

### Section 5: Results

- [ ] Quantitative results (tables)
- [ ] Qualitative analysis (figures)
- [ ] What worked well
- [ ] Limitations and failure modes

### Section 6: Discussion

- [ ] Key insights
- [ ] Comparison to related work
- [ ] Why this approach is interesting
- [ ] Future directions (ARC-AGI, robotics)

### Section 7: Conclusion

- [ ] Summary of contributions
- [ ] Broader impact
- [ ] Next steps

### Appendix

- [ ] Full hyperparameter tables
- [ ] Additional results
- [ ] Code availability
- [ ] Reproducibility details

## 🖥️ Code & Reproducibility

- [ ] Clean up code
  - Remove debug prints
  - Add docstrings
  - Consistent style
  
- [ ] Create reproducibility scripts
  - [ ] `scripts/reproduce_baseline.sh`
  - [ ] `scripts/reproduce_eqm.sh`
  - [ ] `scripts/reproduce_all.sh`

- [ ] Document environment
  - [ ] `requirements.txt` with exact versions
  - [ ] Python version
  - [ ] Hardware specs used

- [ ] Upload to HuggingFace
  - [ ] Trained models
  - [ ] Datasets
  - [ ] Model cards with usage examples

- [ ] GitHub release
  - [ ] Tag version (v1.0.0)
  - [ ] Release notes
  - [ ] Link to technical report

## 🎥 Video Content

- [ ] Record training runs
- [ ] Capture gameplay from trained models
- [ ] Screen recording of energy landscape visualization
- [ ] Create compilation for YouTube video

## 📅 Timeline Estimate

### Week 1: Initial Experiments

- Generate datasets
- Train baseline and initial EqM model
- Quick sanity checks

### Week 2: Ablations

- Gradient schedules
- Sampling steps
- Architecture variants

### Week 3: Evaluation

- In-game performance testing
- Metric collection
- Failure analysis

### Week 4: Analysis & Visualization

- Create all figures
- Statistical analysis
- Qualitative insights

### Week 5: Writing

- Draft all sections
- Technical accuracy check
- Clarity pass

### Week 6: Polish

- Final experiments to fill gaps
- Figure refinement
- Proofreading
- Code cleanup

### Week 7: Release

- Upload models to HuggingFace
- Publish report on website
- GitHub release
- YouTube video
- Social media announcement

## 🎯 Success Criteria

Minimum viable technical report needs:

- [ ] At least one fully trained EqM model
- [ ] Comparison to baseline policy
- [ ] 2-3 ablation studies
- [ ] Quantitative results (accuracy, speed)
- [ ] Qualitative analysis (visualizations)
- [ ] Working code released
- [ ] Models available on HuggingFace

Nice to have:

- [ ] In-game performance evaluation
- [ ] Multiple architecture variants
- [ ] Energy landscape visualization
- [ ] Comprehensive ablations
- [ ] Video demos

## 📚 References to Gather

- [ ] Original EqM paper (Wang & Du)
- [ ] Flow Matching papers (for comparison)
- [ ] World models literature (Ha & Schmidhuber, Hafner et al.)
- [ ] Energy-Based Models (LeCun, Hinton)
- [ ] Related work in model-based RL
- [ ] Snake RL baselines (if any)

## 🐛 Known Issues to Address

Before publishing:

- [ ] Test dataset generation on different seeds
- [ ] Verify next_state computation is correct
- [ ] Check for memory leaks in training loop
- [ ] Ensure reproducibility across different GPUs
- [ ] Test on different hardware (CPU-only, single GPU, multi-GPU)

## 💡 Optional Extensions

If time permits and results are good:

- [ ] Multi-step prediction (predict t+2, t+3)
- [ ] Uncertainty quantification (gradient norm as confidence)
- [ ] Planning via trajectory optimization
- [ ] Transfer to other grid-world games
- [ ] Preliminary ARC-AGI experiments

---

## 📝 Notes

- Keep track of all experiment configs in W&B or similar
- Save random seeds for reproducibility
- Document any unexpected findings
- Take screenshots of interesting behaviors
- Write as you go - don't wait until the end

## ✉️ Feedback Loop

As you run experiments:

1. Update this checklist
2. Note surprising results
3. Adjust plans based on findings
4. Iterate quickly on promising directions
5. Don't get stuck perfecting one experiment

The goal is **insight**, not perfection. Ship the report when you have something interesting to say, even if not all ablations are complete.

---

**Last Updated**: 18 October 2025  
**Status**: Planning phase - no experiments run yet
