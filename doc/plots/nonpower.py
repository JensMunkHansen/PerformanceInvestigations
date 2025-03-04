import matplotlib.pyplot as plt
import numpy as np

# Define data with separate groups for serial and parallel execution times
sizes = ["512", "1024", "1536", "528", "1040", "1552"]
categories = ["Baseline", "Non-power-of-2"]

serial_times = [
    [114, 121],   # 512 / 528
    [1921, 954],  # 1024 / 1040
    [6478, 3211]  # 1536 / 1552
]

parallel_times = [
    [12.3, 12.0],  # 512 / 528
    [172, 105],    # 1024 / 1040
    [514, 299]     # 1536 / 1552
]

# Create new groups to separate serial and parallel completely
group_labels = ["Serial 512/528", "Serial 1024/1040", "Serial 1536/1552", "Parallel 512/528", "Parallel 1024/1040", "Parallel 1536/1552"]
x = np.arange(len(group_labels))
width = 0.25

# Combine serial and parallel times into separate lists for better grouping
serial_values = serial_times + [[0, 0], [0, 0], [0, 0]]  # Padding for alignment
parallel_values = [[0, 0], [0, 0], [0, 0]] + parallel_times  # Padding for alignment

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot Serial times on left axis
for i, cat in enumerate(categories):
    ax1.bar(x[:3] + i * width - width / 2, [row[i] for row in serial_values[:3]], width, label=f"{cat} (Serial)")

# Plot Parallel times on right axis
for i, cat in enumerate(categories):
    ax2.bar(x[3:] + i * width - width / 2, [row[i] for row in parallel_values[3:]], width, label=f"{cat} (Parallel)", alpha=0.7)

# Labels and legends
ax1.set_xlabel("Task")
ax1.set_ylabel("Serial Execution Time (ms)", color="blue")
ax2.set_ylabel("Parallel Execution Time (ms)", color="red")

ax1.set_xticks(x)
ax1.set_xticklabels(group_labels, rotation=20)

# Adjust legends to fit within the figure
ax1_legend = ax1.legend(loc="upper left", bbox_to_anchor=(0.75, 1))
ax2_legend = ax2.legend(loc="upper left", bbox_to_anchor=(0.75, 0.85))
fig.add_artist(ax1_legend)  # Ensure both legends are properly placed

plt.title("Execution Times Comparison (Baseline vs Non-power-of-2)")
plt.tight_layout()
plt.show()
