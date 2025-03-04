import matplotlib.pyplot as plt
import numpy as np

# Data
tasks = ["Serial 512", "Serial 1024", "Serial 1536", "Par 512", "Par 1024", "Par 1536"]
languages = ["C++", "C#", "AOT"]

serial_times = [
    [114, 265, 274],   # Serial 512
    [1921, 2353, 2337], # Serial 1024
    [6478, 7942, 7812]  # Serial 1536
]

parallel_times = [
    [12.3, 33.88, 32.34],  # Par 512
    [172, 280, 266],       # Par 1024
    [514, 1488, 925]       # Par 1536
]

# Bar positions
x = np.arange(len(tasks))
width = 0.25

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot Serial times
for i, lang in enumerate(languages):
    ax1.bar(x[:3] + i * width - width, [row[i] for row in serial_times], width, label=f"{lang} (Serial)")

# Plot Parallel times
for i, lang in enumerate(languages):
    ax2.bar(x[3:] + i * width - width, [row[i] for row in parallel_times], width, label=f"{lang} (Parallel)", alpha=0.7)

# Labels and legends
ax1.set_xlabel("Task")
ax1.set_ylabel("Serial Execution Time (ms)", color="blue")
ax2.set_ylabel("Parallel Execution Time (ms)", color="red")

ax1.set_xticks(x)
ax1.set_xticklabels(tasks, rotation=20)

ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
ax2.legend(loc="upper right", bbox_to_anchor=(1.02, 0.6))

plt.title("Execution Times Comparison (C++ vs C# vs AOT)")
plt.tight_layout()
plt.show()
