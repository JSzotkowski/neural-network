from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import json
import numpy as np

# Load your JSON data (replace 'data.json' with your JSON file)
with open('/home/jiri/PycharmProjects/NeuralNetwork/terrain_to_visualize.json', 'r') as json_file:
    data = json.load(json_file)

print(len(data))

x_values = []
y_values = []
f_values = []

for row in data:
    for col in row:
        x, y, f = col
        x_values.append(x)
        y_values.append(y)
        f_values.append(f)

# Convert the lists to NumPy arrays for easier manipulation
x_values = np.array(x_values)
y_values = np.array(y_values)
f_values = np.array(f_values)

# Define the range of x and y values based on your data
x_range = np.linspace(min(x_values), max(x_values), num=len(data))
y_range = np.linspace(min(y_values), max(y_values), num=len(data))

# Create a grid of (x, y) points
xx, yy = np.meshgrid(x_range, y_range)

# Interpolate the function values on the grid
interpolated_f = griddata((x_values, y_values), f_values, (xx, yy), method='linear')

# Create a contour plot
plt.contourf(xx, yy, interpolated_f, cmap='viridis')
plt.colorbar(label='f(x, y)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour Plot of f(x, y)')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, interpolated_f, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
ax.set_title('Surface Plot of f(x, y)')
plt.show()
