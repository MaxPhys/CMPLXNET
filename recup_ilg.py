import numpy as np
import matplotlib.pyplot as plt


# Define the creation of the small-world network using the Watt-Strogatz algorithm
def watts_strogatz_model(N, k, p):
    # Create a regular ring lattice
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(1, k // 2 + 1):
            G[i, (i+j) % N] = 1
            G[i, (i-j) % N] = 1

    # Rewire the edges with rewiring probability p
    for i in range(N):
        for j in range(1, k // 2 + 1):
            if np.random.rand() < p:
                new_j = np.random.randint(0, N)
                while new_j == i or G[i, new_j] == 1:
                    new_j = np.random.randint(0, N)
                G[i, (i+j) % N] = 0
                G[(i+j) % N, i] = 0
                G[i, new_j] = 1
                G[new_j, i] = 1

    return G


# Define the function for measuring the clustering coefficient, as explained in the assigment
def measure_clustering_coefficient(G):
    N = len(G)
    num_triangles = 0
    num_connected_triplets = 0

    for i in range(N):
        neighbors = np.nonzero(G[i])[0]
        num_connected_triplets += len(neighbors) * (len(neighbors) - 1) // 2
        for j in neighbors:
            common_neighbors = np.nonzero(G[j] & G[i])[0]
            num_triangles += len(common_neighbors)

    if num_connected_triplets == 0:
        return 0
    else:
        return 3.0 * num_triangles / num_connected_triplets


# Define the function to inizialize the states of the networks nodes
def initialize_states(N, initial_fraction_infected):
    # Initialize the state of nodes (0 for susceptible, 1 for infected)
    num_infected = int(initial_fraction_infected * N)
    states = np.zeros(N)
    infected_indices = np.random.choice(N, num_infected, replace=False)
    states[infected_indices] = 1
    return states


# Define the SIS model dynamics that we run on the WS model
def sis_dynamics(G, initial_states, lambda_val, delta, num_steps):
    N = len(G)
    states = initial_states.copy()
    frac_infected = []
    infected_nodes = np.nonzero(states)[0]
    active_links = []

    for step in range(num_steps):
        new_states = states.copy()
        Ni = len(infected_nodes)
        Ea = len(active_links)

        for node in infected_nodes:
            neighbors = np.nonzero(G[node])[0]
            num_infected_neighbors = np.sum(states[neighbors])

            # Infection rule
            infection_prob = lambda_val * num_infected_neighbors
            if np.random.random() < infection_prob:
                new_infected = np.setdiff1d(neighbors, infected_nodes)
                new_states[new_infected] = 1
                infected_nodes = np.union1d(infected_nodes, new_infected)
                for neighbor in new_infected:
                    active_links.append([node, neighbor])

        for link in active_links:
            node, neighbor = link
            # Recovery rule
            recovery_prob = delta * Ni / (delta * Ni + lambda_val * Ea)
            if np.random.random() < recovery_prob:
                new_states[node] = 0
                infected_nodes = np.setdiff1d(infected_nodes, node)
                active_links.remove(link)

        states = new_states.copy()
        frac_infected.append(Ni / N)

    return frac_infected


# Define the function for the steady state prevalence
def steady_state_prevalence(G, initial_states, lambda_values, delta, num_steps):
    N = len(G)
    p_values = [0, 0.5, 1]
    steady_state_prevalence_values = []

    for p in p_values:
        print(f"Calculating steady-state prevalence for p = {p}...")
        prevalence_values = []

        for lambda_val in lambda_values:
            frac_infected = sis_dynamics(G, initial_states, lambda_val, delta, num_steps)
            steady_state_prevalence = np.mean(frac_infected[-int(num_steps / 2):])  # Using the last half as steady state
            prevalence_values.append(steady_state_prevalence)

        steady_state_prevalence_values.append(prevalence_values)

    return steady_state_prevalence_values


# Main function definition to execute
def main():
    N = 10**4  # Number of nodes (given)
    k = 4      # Average degree (given)
    c0 = 0.25  # Initial fraction of infected nodes
    lambda_values = [0, 0.01, 0.1, 0.5, 1]
    delta = 1
    num_steps = 100

    # Measure clustering coefficient for p = 0, 0.5, 1
    clustering_coefficients = []
    for p in [0, 0.5, 1]:
        G = watts_strogatz_model(N, k, p)
        clustering_coefficient = measure_clustering_coefficient(G)
        clustering_coefficients.append(clustering_coefficient)

    # Print clustering coefficient results
    for p, cc in zip([0, 0.5, 1], clustering_coefficients):
        print(f"Clustering coefficient for p = {p}: {cc}")

    # Plot the clustering coefficients in one plot
    plt.figure(figsize=(8, 6))
    plt.plot([0, 0.5, 1], clustering_coefficients, marker='s', linestyle='-', color='b')
    plt.xlabel('Rewiring Probability (p)')
    plt.ylabel('Clustering Coefficient C(p)')
    plt.title('C(p) as a Function of p in WS Model')
    plt.grid(True)
    plt.show()

    # Plot the SIS dynamics for different lambda values in three separate plots, each one for one of the given rewiring
    # probabilities that were given in the task
    num_colors = len(lambda_values)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

    for p in [0, 0.5, 1]:  #
        G = watts_strogatz_model(N, k, p)
        initial_states = initialize_states(N, c0)

        plt.figure(figsize=(10, 6))
        labels = [f'λ = {lambda_val}' for lambda_val in lambda_values]

        for i, lambda_val in enumerate(lambda_values):
            frac_infected = sis_dynamics(G, initial_states, lambda_val, delta, num_steps)
            plt.plot(range(num_steps), frac_infected, color=colors[i % num_colors], label=labels[i])

        plt.xlabel('Time Steps')
        plt.ylabel('Fraction of Infected Nodes')
        plt.title(f'SIS Dynamics on WS Model (p = {p})')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot the steady-state prevalence as a function of lambda for all three rewiring probabilities
    steady_state_prevalence_values = steady_state_prevalence(G, initial_states, lambda_values, delta, num_steps)

    plt.figure(figsize=(8, 6))
    for i, p in enumerate([0, 0.5, 1]):
        plt.plot(lambda_values, steady_state_prevalence_values[i], marker='o', linestyle='-', label=f'p = {p}')

    plt.xlabel('Infection Rate (λ)')
    plt.ylabel('Steady-State Prevalence')
    plt.title('Steady-State Prevalence as a Function of λ in WS Model')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
