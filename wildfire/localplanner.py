import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def smooth_path(x, y, smoothing_factor=0.5):
    n_x=[]
    n_y=[]
    
    for i in len(x)-1:
        x_s= np.linspace(x[i], x[i+1])
        y_s= np.linspace(y[i], y[i+1])
        for i in range(len(x_s)):
            n_x.append(x_s[i])
            n_y.append(y_s[i])
            
    
    
    
    # Create a parameterization of the path
    t = np.arange(len(n_x))

    # Fit a cubic spline to the data
    cs_x = CubicSpline(t, n_x, bc_type='clamped')
    cs_y = CubicSpline(t, n_y, bc_type='clamped')

    # Evaluate the spline at a finer set of points
    t_fine = np.linspace(t.min(), t.max(), 10 * (len(t) - 1))
    path_smooth_x = cs_x(t_fine)
    path_smooth_y = cs_y(t_fine)

    return path_smooth_x, path_smooth_y

def calculate_curvature(x, y):
    dx_dt = np.gradient(x)
    dy_dt = np.gradient(y)
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)

    curvature = np.abs((dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) / (dx_dt**2 + dy_dt**2)**1.5)

    return curvature

def filter_by_curvature(x, y, max_curvature):
    curvature = calculate_curvature(x, y)

    # Keep points with curvature below the threshold
    filtered_x = [x[i] for i in range(len(x)) if curvature[i] < max_curvature]
    filtered_y = [y[i] for i in range(len(y)) if curvature[i] < max_curvature]

    return np.array(filtered_x), np.array(filtered_y)

# Example path points
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 3, 1, 4, 2, 5])

# Smooth the path
smoothed_x, smoothed_y = smooth_path(x, y)

# Filter points based on curvature
max_curvature = 0.1  # You can adjust this threshold based on your robot's turning capabilities
filtered_x, filtered_y = filter_by_curvature(smoothed_x, smoothed_y, max_curvature)

# Plot the original, smoothed, and filtered paths
plt.plot(x, y, 'o-', label='Original Path')
plt.plot(smoothed_x, smoothed_y, 's-', label='Smoothed Path')
plt.plot(filtered_x, filtered_y, '^-', label='Filtered Path')
plt.legend()
plt.grid(True)
plt.show()
