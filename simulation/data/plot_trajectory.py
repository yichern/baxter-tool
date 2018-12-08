from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import glob

#READ IN FILES:

#from folder, grab files, for each file,  then concatenate together with one column for color and another for type of marker

#files = sorted(glob.glob("11-baxter-paths/*"))
files = sorted(glob.glob("trpo-baxter-11-paths/*"))
files = sorted(glob.glob("ddpg-baxter-11-2-paths/*"))

all_files = []

for i in range(len(files)):

	with open(files[i]) as f:
	    content = f.readlines()

	content = [x.strip() for x in content] 

	for line in content:
		array = line.split(' ')
		points = np.array(array[:3]).astype(np.float)
		if i != 10:
			a = points.tolist() + [i, 'b', 'o']
			all_files.append(a)
		else:
			a = points.tolist() + [i, 'r', '^']
			all_files.append(a)

	#print(array[:3])

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)

for x, y, z, i, c, m in all_files:
	ax.scatter(x,y,z,c=c)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()