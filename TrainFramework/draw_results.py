##绘制多个条形图
from matplotlib import pyplot as plt

fig = plt.figure()

ax = fig.add_subplot(111, title="abcd")
ax.set_ylabel('N')
ax.set_xlabel('S')

a = ["DLevel 1","DLevel 2","DLevel 3","DLevel 4", "FD"]

b_14 = [99.3, 79.7, 71.1, 74.8, 13.1]
b_15 = [99.1, 67.6, 45.1, 22.5, 4.4]
b_16 = [99.3, 78.0, 71.7, 86.5, 5.3]

bar_width=0.3
x_14 = list(range(len(a)))
x_15 = [i+bar_width for i in x_14]
x_16 = [i+bar_width for i in x_15]

ax.bar(x_14, b_14, width=0.3, label="Normal")
ax.bar(x_15, b_15, width=0.3, label="SS")
ax.bar(x_16, b_16, width=0.3, label="SSP-SPL-BC")
# ##设置x轴刻度
plt.xticks(x_15,a)
ax.legend()

fig.savefig("pic_test.png")