#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# your code here

# main specs
fig = plt.figure(figsize=(6, 6))
fig.suptitle("All in one")
grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.5)

# linear plot
ax1 = fig.add_subplot(grid[0, 0])
plt.xlim(0, 10)
plt.plot(y0, color='red')

# scatter plot
ax2 = fig.add_subplot(grid[0, 1])
plt.xlabel('Height (in)', fontsize='x-small')
plt.ylabel('Weight (lbs)', fontsize='x-small')
plt.title("Men's Height vs Weight", fontsize='x-small')
plt.scatter(x1, y1, c='magenta')

# exponential log plot
ax3 = fig.add_subplot(grid[1, 0])
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title("Exponential Decay of C-14", fontsize='x-small')
plt.yscale("log")
plt.margins(0, 28650)
plt.plot(x2, y2)

# exponential plot
ax4 = fig.add_subplot(grid[1, 1])
plt.xlabel('Time (years)', fontsize='x-small')
plt.ylabel('Fraction Remaining', fontsize='x-small')
plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
plt.xlim(0, 20000)
plt.ylim(0, 1)
plt.plot(x3, y31, c='red', linestyle='dashed', label='C-14')
plt.plot(x3, y32, c='green', label='Ra-226')
plt.legend()


# histogram plot
ax5 = fig.add_subplot(grid[2, :2])
bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.xlabel('Grades', fontsize='x-small')
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.ylabel('Number of Students', fontsize='x-small')
plt.title('Project A', fontsize='x-small')
plt.hist(student_grades, bins, edgecolor='black')
plt.xticks(np.arange(0, 110, 10))

plt.show()
