import matplotlib.pyplot as plt
import numpy as np

# Data for MAE, RMSE, and MAPE
x_labels = ['Point 1', 'Point 2', 'Point 3', 'Point 4', 'Point 5', 'Point 6']
MAE_values = [50.03, 46.60, 27.23, 30.75, 29.60, 28.91]
RMSE_values = [101.27, 79.43, 59.37, 61.11, 62.03, 56.37]
MAPE_values = [0.20, 0.20, 0.11, 0.18, 0.13, 0.14]

# Set up the figure and style
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()

# Plot each error metric with a specific marker and line style
ax.plot(range(1, 7), MAE_values, 'x', markeredgewidth=2, label='MAE')
ax.plot(range(1, 7), RMSE_values, '-', linewidth=2, label='RMSE')
ax.plot(range(1, 7), MAPE_values, 'o-', linewidth=2, label='MAPE')

# Customize the axes
ax.set(xlim=(0.5, 5.5), xticks=np.arange(1, 7), xticklabels=x_labels,
       ylim=(0, 120), yticks=np.arange(0, 121, 20))

# Add legend and labels
ax.set_xlabel('Data Points')
ax.set_ylabel('Error Values')
ax.set_title('Error Metrics Comparison')
ax.legend()

plt.show()
