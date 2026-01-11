import matplotlib.pyplot as plt
import numpy as np

# Data extracted from Training Logs (Head and Tail)
steps_head = [10, 20, 30, 40]
beta_head = [0.7815, 0.7687, 0.7811, 0.7821]

steps_tail = [4910, 4920, 4930, 4940, 4950, 4960, 4970, 4980, 4990, 5000]
beta_tail = [0.8583, 0.9186, 0.8667, 0.8830, 0.9148, 0.8588, 0.8824, 0.8772, 0.8871, 0.8888]

# Create figure
plt.figure(figsize=(10, 6))

# Plot Head
plt.plot(steps_head, beta_head, 'b-o', label='Early Adaptation (High Surprise)')

# Plot Tail
plt.plot(steps_tail, beta_tail, 'g-o', label='Late Consolidation (Stability)')

# Connect with dashed line to imply trajectory
plt.plot([40, 4910], [0.7821, 0.8583], 'k--', alpha=0.3, label='Interim Training')

plt.title(r"Neuromodulatory Plasticity ($\beta$) Dynamics")
plt.xlabel("Training Steps")
plt.ylabel("Beta (Predictive Coding Weight)")
plt.ylim(0.5, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()

# Annotations
plt.annotate('Initial Shock\n(Beta drops to ~0.78)', xy=(25, 0.775), xytext=(200, 0.65),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.annotate('Recovery\n(Beta restores to ~0.89)', xy=(4950, 0.89), xytext=(4000, 0.95),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.savefig('beta_dynamics.png')
print(">> Saved beta_dynamics.png")
