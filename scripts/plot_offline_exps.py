import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

operators = [4, 8, 12]
minutes = [3.75, 7.5]

mse = np.array([
    [0.27092, 0.25969],
    [0.26932, 0.73342],
    [0.26050, np.nan],
])

plt.figure(figsize=(6, 4))
sns.heatmap(
    mse,
    annot=True,
    fmt=".3f",
    cmap="viridis",
    xticklabels=minutes,
    yticklabels=operators,
    cbar_kws={"label": "MSE"}
)

plt.xlabel("Minutes per Scene")
plt.ylabel("# Operators")
plt.title("MSE Avg In-Domain Operators")
plt.tight_layout()

# SAVE
plt.savefig("mse_heatmap.png", dpi=300, bbox_inches="tight")
# plt.savefig("mse_heatmap.pdf", bbox_inches="tight")  # optional, for papers

plt.show()
