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

# Compute speed-up factors relative to the Baseline Serial execution times (Baseline Serial = 1.0)
reference_baseline_serial = serial_times["Baseline"]  # The baseline for speed-up calculations

speedup_serial = {category: [ref / value for ref, value in zip(reference_baseline_serial, values)]
                  for category, values in serial_times.items()}

speedup_parallel = {category: [ref / value if value is not None else None
                               for ref, value in zip(reference_baseline_serial, values)]
                    for category, values in parallel_times.items()}

# Convert data into lists for plotting
x = np.arange(len(sizes) * 2)  # Double the groups to separate serial and parallel
width = 0.12  # Width of bars

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot serial speed-up factors
for i, (category, values) in enumerate(speedup_serial.items()):
    ax1.bar(x[:3] + i * width - width * 3, values, width, label=f"{category} (Serial Speed-up)")

# Plot parallel speed-up factors
for i, (category, values) in enumerate(speedup_parallel.items()):
    filtered_values = [v if v is not None else 0 for v in values]  # Replace None with 0
    ax2.bar(x[3:] + i * width - width * 3, filtered_values, width, label=f"{category} (Parallel Speed-up)", alpha=0.7)

# Labels and legends
ax1.set_xlabel("Task")
ax1.set_ylabel("Speed-up Factor (Baseline Serial = 1.0)", color="blue")
ax2.set_ylabel("Speed-up Factor (Baseline Serial = 1.0)", color="red")

ax1.set_xticks(x)
ax1.set_xticklabels(["Serial 512", "Serial 1024", "Serial 1536", "Parallel 512", "Parallel 1024", "Parallel 1536"], rotation=20)

# Adjust legends within the figure
ax1_legend = ax1.legend(loc="upper left", bbox_to_anchor=(0.65, 1))
ax2_legend = ax2.legend(loc="upper left", bbox_to_anchor=(0.65, 0.75))
fig.add_artist(ax1_legend)

plt.title("Speed-up Comparison (Reference: Baseline Serial = 1.0)")
plt.tight_layout()
plt.show()
