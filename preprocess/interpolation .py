def interpolate_points(a, b, c, d, n):
    # Initialize lists to store the interpolated points
    x_interpolated = []
    y_interpolated = []

    # Calculate the step size for interpolation
    step_x = (c - a) / (n + 1)
    step_y = (d - b) / (n + 1)

    # Interpolate n points
    for i in range(1, n + 1):
        x_interpolated.append(a + i * step_x)
        y_interpolated.append(b + i * step_y)

    return x_interpolated, y_interpolated

# Example usage:
a, b =-250.727142,-3770.291504
c, d =-242.228088,-3767.773682
n = 2
x_interpolated, y_interpolated = interpolate_points(a, b, c, d, n)

print("Interpolated Points:")
for x, y in zip(x_interpolated, y_interpolated):
    print(f"({x}, {y})")
