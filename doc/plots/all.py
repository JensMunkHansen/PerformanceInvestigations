import matplotlib.pyplot as plt
import numpy as np

# Define categories and data
sizes = ["512", "1024", "1536"]
categories = ["Baseline", "B", "AB", "BC", "ABC", "BCMO", "MKL"]

# Serial execution times
serial_times = {
    "Baseline": [114, 1921, 6478],
    "B": [22.4, 241, 744],
    "AB": [19.7, 166, 558],
    "BC": [19.6, 170, 563],
    "ABC": [19.7, 164, 550],
    "BCMO": [5.32, 62.7, 151]
}

# Parallel execution times (including MKL)
parallel_times = {
    "Baseline": [12.3, 172, 514],
    "B": [None, None, None],  # No parallel data given
    "AB": [2.66, 37.3, 125],
    "BC": [2.92, 32.6, 117],
    "ABC": [None, None, None],  # No parallel data given
    "BCMO": [2.61, 28.7, 76.1],
    "MKL": [0.514, 3.99, 13.0]  # MKL placed in parallel execution
}

# Convert data into lists for plotting
x = np.arange(len(sizes) * 2)  # Double the groups to separate serial and parallel
width = 0.12  # Width of bars

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot serial execution times
for i, (category, values) in enumerate(serial_times.items()):
    ax1.bar(x[:3] + i * width - width * 3, values, width, label=f"{category} (Serial)")

# Plot parallel execution times
for i, (category, values) in enumerate(parallel_times.items()):
    filtered_values = [v if v is not None else 0 for v in values]  # Replace None with 0
    ax2.bar(x[3:] + i * width - width * 3, filtered_values, width, label=f"{category} (Parallel)", alpha=0.7)

# Labels and legends
ax1.set_xlabel("Task")
ax1.set_ylabel("Serial Execution Time (ms)", color="blue")
ax2.set_ylabel("Parallel Execution Time (ms)", color="red")

ax1.set_xticks(x)
ax1.set_xticklabels(["Serial 512", "Serial 1024", "Serial 1536", "Parallel 512", "Parallel 1024", "Parallel 1536"], rotation=20)

# Adjust legends within the figure
ax1_legend = ax1.legend(loc="upper left", bbox_to_anchor=(0.65, 1))
ax2_legend = ax2.legend(loc="upper left", bbox_to_anchor=(0.65, 0.75))
fig.add_artist(ax1_legend)

plt.title("Execution Times Comparison (Various Methods)")
plt.tight_layout()
plt.show()
