# Pasture = [306, 624, 370, 3914, 1074, 3115, 3734, 4077, 2174, 1088]
# # Permanent_crops = [506, 776, 556, 3699, 1148, 2879, 3688, 3900, 1934, 1049]
# Forests = [197, 401, 268, 3046, 738, 2406, 2976, 3266, 1582, 747]
# # channel_name = ['Blue', 'Green', 'Red', 'NIR', 'Red edge 1', 'Red edge 2', 'Red edge 3', 'Red edge 4', 'SWIR 1', 'SWIR 2']
# channel_name = ['490', '560', '665', '842', '705', '740', '782', '865', '1610', '2190']

# import matplotlib.pyplot as plt

# # # 数据
# # Pasture = [377, 735, 328, 5012, 1170, 4055, 4954, 5176, 2174, 1036]
# # Permanent_crops = [506, 776, 556, 3699, 1148, 2879, 3688, 3900, 1934, 1049]
# # channel_name = ['Blue', 'Green', 'Red', 'NIR', 'Red edge 1', 'Red edge 2',
# #                 'Red edge 3', 'Red edge 4', 'SWIR 1', 'SWIR 2']

# # 画图
# plt.figure(figsize=(6, 4))
# plt.plot(channel_name, Pasture, marker='o', label='Pasture')
# plt.plot(channel_name, Forests, marker='s', label='Forests')

# # 样式
# # plt.title('Spectral Signature Comparison')
# plt.xlabel('Central wavelength (nm)')
# plt.ylabel(r'Mean reflectance $\times 10^4$')
# # plt.xticks(rotation=45, ha='right') 
# plt.legend()
# # plt.grid(True)
# plt.tight_layout()

# # 保存到本地
# plt.savefig('spectral_signature_1986.png', dpi=300)

# # 可选：显示图像
# # plt.show()



import matplotlib.pyplot as plt

# 原始数据
Pasture = [306, 624, 370, 3914, 1074, 3115, 3734, 4077, 2174, 1088]
Forests = [197, 401, 268, 3046, 738, 2406, 2976, 3266, 1582, 747]
channel_name = ['490', '560', '665', '842', '705', '740', '782', '865', '1610', '2190']

# 将842放到红边区间之后（排序）
sort_idx = [0, 1, 2, 4, 5, 6, 3, 7, 8, 9]  # 对应 490–2190 按波长视觉顺序
Pasture_sorted = [Pasture[i] for i in sort_idx]
Forests_sorted = [Forests[i] for i in sort_idx]
labels_sorted = [channel_name[i] for i in sort_idx]
x = list(range(len(labels_sorted)))

# 画图
plt.figure(figsize=(7, 4))
plt.plot(x, Pasture_sorted, marker='o', label='Pasture')
plt.plot(x, Forests_sorted, marker='s', label='Forests')

# 坐标轴与标签设置
plt.xticks(ticks=x, labels=labels_sorted)
plt.xlabel('Central wavelength (nm)')
plt.ylabel(r'Mean reflectance $\times 10^4$')
plt.legend()
plt.tight_layout()

# 保存图像
plt.savefig('spectral_signature_pasture_vs_forests_1986.png', dpi=300)
plt.show()
