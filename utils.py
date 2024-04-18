import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

# 时间参数
dt = 0.1  # 时间间隔
T = 80    # 总时间步数 (8s / 0.1s)

# 车辆参数（简化模型）
L = 2.0  # 车辆轴距，影响转向
initial_state = jnp.array([0.0, 0.0, 0.0, 11.1])


def update_state(state, control, dt):
    """
    更新车辆状态，包括位置、朝向和速度。

    参数:
    state - 当前状态，格式为 [x, y, theta, v]
    control - 控制输入，格式为 [a, steer]
    dt - 时间步长
    """
    x, y, theta, v = state
    a, steer = control

    # 更新速度
    v_next = v + a * dt

    # 更新朝向
    theta_next = theta + (v / L) * jnp.tan(steer) * dt

    # 更新位置
    x_next = x + v * jnp.cos(theta) * dt
    y_next = y + v * jnp.sin(theta) * dt

    return jnp.array([x_next, y_next, theta_next, jnp.clip(v_next, 0.)])


def simulate_trajectory(initial_state, controls, dt=dt):
    """
    模拟给定控制输入下的整个轨迹。

    参数:
    initial_state - 初始状态，格式为 [x, y, theta, v]
    controls - 控制序列，每一行是一个时间步的 [a, steer]
    dt - 时间步长
    """
    states = [initial_state]
    state = initial_state
    for control in controls.reshape(-1, 2):
        state = update_state(state, control, dt)
        states.append(state)
    return jnp.stack(states)


def cost_function(control_points):
    controls = calculate_controls(control_points)
    trajectory = simulate_trajectory(initial_state, controls)
    target = jnp.array([88.8, 3.75, 0.0, 11.1])  # 目标位置
    final_position = trajectory[-1, :]
    return 10 * jnp.sum((final_position - target)**2) + 0.1 * jnp.sum(controls[:, 1]**2) + 0.1 * jnp.sum(controls[:, 0]**2) + jnp.sum((trajectory[:, 3] - target[3])**2)


def bezier_curve(control_points, t):
    """ 计算五阶Bezier曲线在t时刻的点 """
    # 解构控制点
    P0, P1, P2, P3, P4, P5 = control_points.reshape(-1, 2)
    # 计算Bezier曲线位置
    B_t = ((1-t)**5)*P0 + 5*t*((1-t)**4)*P1 + 10*(t**2)*((1-t)**3)*P2 + \
        10*(t**3)*((1-t)**2)*P3 + 5*(t**4)*(1-t)*P4 + (t**5)*P5
    return B_t


def calculate_controls(control_points, dt=0.1):
    num_steps = int(8 / dt)  # 总时间步长
    t_values = jnp.linspace(0, 1, num_steps)  # 参数t的值
    points = jax.vmap(lambda t: bezier_curve(
        control_points, t))(t_values)  # 曲线上的点

    # 计算切线（速度向量）
    velocities = jnp.gradient(points, axis=0) / dt
    speeds = jnp.linalg.norm(velocities, axis=1)
    directions = jnp.arctan2(velocities[:, 1], velocities[:, 0])

    # 计算加速度
    accelerations = jnp.gradient(speeds) / dt

    # 计算转向（方向的导数）
    steers = jnp.gradient(directions) / dt

    return jnp.column_stack((accelerations, steers))


def plot_trajectory(trajectory):
    x_positions = trajectory[:, 0]
    y_positions = trajectory[:, 1]

    plt.figure(figsize=(8, 6))
    plt.plot(x_positions, y_positions, marker='o',
             color='skyblue', label='Trajectory')
    plt.scatter(x_positions[0], y_positions[0],
                color='green', s=100, label='Start Point')
    plt.scatter(x_positions[-1], y_positions[-1],
                color='red', s=100, label='End Point')
    plt.title("Vehicle Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    # Ensure the scale of x and y axes are equal to better visualize the trajectory
    plt.axis("equal")
    plt.show()


def plot_trajectory_details(trajectory, controls, dt=dt):
    num_steps = trajectory.shape[0]
    t = jnp.linspace(0, num_steps * dt, num_steps)

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    theta = trajectory[:, 2]
    v = trajectory[:, 3]
    a = controls[:, 0]
    steer = controls[:, 1]

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # 3行2列

    # y vs x
    axs[0, 0].plot(x, y, 'b-')
    axs[0, 0].set_title("Trajectory y(x)")
    axs[0, 0].set_xlabel("x position")
    axs[0, 0].set_ylabel("y position")
    axs[0, 0].axis('equal')

    # x vs t
    axs[1, 0].plot(t, x, 'r-')
    axs[1, 0].set_title("x position over time")
    axs[1, 0].set_xlabel("time (s)")
    axs[1, 0].set_ylabel("x position")

    # y vs t
    axs[0, 1].plot(t, y, 'g-')
    axs[0, 1].set_title("y position over time")
    axs[0, 1].set_xlabel("time (s)")
    axs[0, 1].set_ylabel("y position")

    # v vs t
    axs[1, 1].plot(t, v, 'c-')
    axs[1, 1].set_title("Velocity over time")
    axs[1, 1].set_xlabel("time (s)")
    axs[1, 1].set_ylabel("Velocity")

    # a vs t
    axs[0, 2].plot(t[:-1], a, 'm-')  # a is one less than t
    axs[0, 2].set_title("Acceleration over time")
    axs[0, 2].set_xlabel("time (s)")
    axs[0, 2].set_ylabel("Acceleration")

    # steer vs t
    axs[1, 2].plot(t[:-1], steer, 'k-')  # steer is one less than t
    axs[1, 2].set_title("Steering angle over time")
    axs[1, 2].set_xlabel("time (s)")
    axs[1, 2].set_ylabel("Steering angle (radians)")

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.show()

# Assuming you have `trajectory` and `controls` from your simulation
# plot_trajectory_details(best_trajectory, best_controls, dt=0.1)
