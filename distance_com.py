# import numpy as np
# import matplotlib.pyplot as plt

# # 加载数据
# cross_attention_distances = np.loadtxt('crossattention_distances.txt')
# self_attention_distances = np.loadtxt('selfattention_distances.txt')

# # 创建图表
# fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)

# # 设置颜色渐变
# color_map_cross = plt.get_cmap('Blues')
# color_map_self = plt.get_cmap('Reds')
# num_points = len(cross_attention_distances)

# # 绘制带有颜色渐变的线段
# for i in range(num_points - 1):
#     # Cross Attention
#     color_cross = color_map_cross((i + 1) / num_points)  # 计算当前深度的颜色
#     ax.plot([i, i + 1], [cross_attention_distances[i], cross_attention_distances[i + 1]],
#             color=color_cross, linestyle='-', marker='o', markersize=8)

#     # Self Attention
#     color_self = color_map_self((i + 1) / num_points)  # 使用另一个颜色映射
#     ax.plot([i, i + 1], [self_attention_distances[i], self_attention_distances[i + 1]],
#             color=color_self, linestyle='--', marker='s', markersize=8)

# # 设置图表标题和标签
# ax.set_xlabel("Depth")
# ax.set_ylabel("Attention distance (px)")
# ax.set_ylim(top=30, bottom=0)
# ax.set_xlim(left=0, right=num_points-1)

# # 添加图例
# ax.legend(['Cross Attention', 'Self Attention'], loc='upper right')

# # 显示网格
# ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# # 保存图像
# plt.savefig('distinct_gradient_line_attention_distance_comparison.png', dpi=150)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 加载数据
cross_attention_distances = np.loadtxt('crossattention_distances.txt')
self_attention_distances = np.loadtxt('selfattention_distances.txt')

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

# 突出显示特定深度区间
highlight_patch = plt.Rectangle((18, 0), 1, 40, color='grey', alpha=0.3, label="Highlighted region")
ax.add_patch(highlight_patch)

# 添加虚线双向箭头标注
for depth in [18, 19]:
    cross_height = cross_attention_distances[depth]
    self_height = self_attention_distances[depth]
    diff = cross_height - self_height
    mid_y = (cross_height + self_height) / 2  # 计算中间位置

    # 添加虚线箭头
    ax.annotate("", xy=(depth, cross_height), xytext=(depth, self_height),
                arrowprops=dict(arrowstyle="<->", linestyle="dashed", color='black', lw=1.5))

    # 在箭头中间标注差值，保留一位小数
    ax.text(depth, mid_y, f'{abs(diff):.1f}', fontsize=10,
            va='center', ha='center', bbox=dict(facecolor='white', edgecolor='none', alpha=0.9))

# 修正图例
ax.legend(handles=[cross_handle, self_handle, highlight_patch], loc='upper left')

# 设置图表标题和标签
ax.set_xlabel("Depth")
ax.set_ylabel("Attention distance (px)")
ax.set_ylim(top=60, bottom=0)
ax.set_xlim(left=0, right=num_points - 1)

# 显示网格
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# 保存图像
plt.savefig('attention_distance_comparison_final_dashed_arrow_updated.png', dpi=150)
# plt.show()












