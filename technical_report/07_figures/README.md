# 07 - Figures

Generate all figures for the technical report including architecture diagrams, training curves, ablation comparisons, and qualitative visualizations.

## Required Scripts

- `plot_architecture.py` - Create architecture diagram (encoder → EqM → sample)
- `plot_training_curves.py` - Training/validation loss and accuracy over time
- `plot_gradient_schedules.py` - Visualize c(γ) functions for different schedules
- `plot_ablations.py` - Comparison plots for all ablation studies
- `plot_energy_landscape.py` - Visualize learned energy landscape
- `plot_sampling_trajectories.py` - Show gradient descent optimization paths
- `plot_qualitative_examples.py` - Side-by-side predicted vs actual states
- `generate_all_figures.py` - Master script to generate all figures
