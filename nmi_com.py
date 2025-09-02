import numpy as np
import matplotlib.pyplot as plt

# 加载数据
cross_attention_distances = np.loadtxt('crossattention_nmi.txt')
self_attention_distances = np.loadtxt('selfattention_nmi.txt')

# 创建图表
fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

# 颜色映射
color_map_cross = plt.get_cmap('Blues')
color_map_self = plt.get_cmap('Reds')
num_points = len(cross_attention_distances)

# 绘制带有颜色渐变的线段
cross_handle, = ax.plot([], [], color='blue', linestyle='-', marker='o', label='Proposed')
self_handle, = ax.plot([], [], color='red', linestyle='--', marker='s', label='SATMAE')

for i in range(num_points - 1):
    color_cross = color_map_cross((i + 1) / num_points)
    color_self = color_map_self((i + 1) / num_points)
    ax.plot([i, i+1], [cross_attention_distances[i], cross_attention_distances[i+1]],
            marker='o', markersize=8, color=color_cross, linestyle='-', linewidth=2)
    ax.plot([i, i+1], [self_attention_distances[i], self_attention_distances[i+1]],
            marker='s', markersize=8, color=color_self, linestyle='--', linewidth=2)

# **去掉 Highlighted Region**
# 删除 ax.add_patch(highlight_patch) 和图例中的 highlight_patch

# # 添加虚线双向箭头标注
# for depth in [18, 19]:
#     cross_height = cross_attention_distances[depth]
#     self_height = self_attention_distances[depth]
#     diff = cross_height - self_height
#     mid_y = (cross_height + self_height) / 2  # 计算中间位置

#     # 添加虚线箭头
#     ax.annotate("", xy=(depth, cross_height), xytext=(depth, self_height),
#                 arrowprops=dict(arrowstyle="<->", linestyle="dashed", color='black', lw=1.5))

#     # 在箭头中间标注差值，保留一位小数
#     ax.text(depth, mid_y, f'{abs(diff):.1f}', fontsize=12,
#             va='center', ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# 修正图例 (去掉 Highlighted Region)
ax.legend(handles=[cross_handle, self_handle], loc='upper left')

# **动态调整 y 轴范围** 适应实际数据
y_max = max(cross_attention_distances.max(), self_attention_distances.max()) * 1.2
ax.set_ylim(bottom=0, top=y_max)

# 设置图表标题和标签
ax.set_xlabel("Depth")
ax.set_ylabel("Normalized MI")
ax.set_xlim(left=0, right=num_points - 1)

# 显示网格
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 保存图像
plt.savefig('NMI_comparison_fixed.png', dpi=150)
plt.show()













