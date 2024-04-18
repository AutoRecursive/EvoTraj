from evosax import CMA_ES, DE, OpenES, DES	
from utils import update_state, simulate_trajectory, cost_function, dt, T, L, plot_trajectory, calculate_controls, initial_state, plot_trajectory_details
import jax.numpy as jnp
from jax import random
import jax

# 80 time steps, each with 2 control variables (acceleration, steer)
num_params = 2 * 6
popsize = 1024
key = random.PRNGKey(0)
strategy = DE(num_dims=num_params, popsize=popsize)
state = strategy.initialize(key)

num_generations = 20
for gen in range(num_generations):
    key, subkey = random.split(key)
    params, state = strategy.ask(subkey, state)
    costs = jax.vmap(cost_function)(params)

    state = strategy.tell(params, costs, state)

    print(f"Generation {gen + 1}: Best Cost = {jnp.min(costs)}")
best_idx = jnp.argmin(costs)
best_controls_points = params[best_idx]
best_control_seq = calculate_controls(best_controls_points)
best_trajectory = simulate_trajectory(
    initial_state, best_control_seq)

print("Best trajectory (x, y, theta):")
print(best_trajectory)

plot_trajectory_details(best_trajectory, best_control_seq)
