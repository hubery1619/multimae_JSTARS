import numpy as np
import matplotlib.pyplot as plt

# 加载数据
cross_attention_distances = np.loadtxt('crossattention_distances.txt')
self_attention_distances = np.loadtxt('selfattention_distances.txt')

# 设置图表尺寸
fig, ax = plt.subplots(figsize=(4.0, 3), dpi=200)

# 绘制折线
cross_handle, = ax.plot(cross_attention_distances, marker='o', linestyle='-', color='blue', label='Proposed')
self_handle, = ax.plot(self_attention_distances, marker='s', linestyle='--', color='red', label='SatMAE')

# **调整图例位置**
ax.legend(loc='upper right', fontsize=8, frameon=True)  # 右上角，避免挡住曲线
# 或者放到图外：
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, frameon=True)  # 放在图外

# 设置标签
ax.set_xlabel("Depth", fontsize=10)
ax.set_ylabel("Attention distance (px)", fontsize=10)

# 调整坐标轴范围
ax.set_xlim(0, len(cross_attention_distances) - 1)
ax.set_ylim(0, 60)

# 网格
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 使图像更紧凑
fig.savefig('attention_distance_comparison_adjusted_legend.png', bbox_inches='tight')
# plt.show()













