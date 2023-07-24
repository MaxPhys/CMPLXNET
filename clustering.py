import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import random
from scipy import stats


def watts_strogatz_model(N, k, p):
    # Create a regular ring lattice
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(1, k // 2 + 1):
            G[i, (i+j) % N] = 1
            G[i, (i-j) % N] = 1

    # Rewire edges with probability p
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


def degree_distribution(G):
    N = len(G)
    degrees = np.sum(G, axis=1)
    degree_counts = np.bincount(degrees)
    degree_probs = degree_counts / N
    return degree_probs


def initialize_states(N, initial_fraction_infected):
    # Initialize the state of nodes (0 for susceptible, 1 for infected)
    num_infected = int(initial_fraction_infected * N)
    states = np.zeros(N)
    infected_indices = np.random.choice(N, num_infected, replace=False)
    states[infected_indices] = 1
    return states


def sis_dynamics(G, initial_states, lambda_val, delta, num_steps):
    N = len(G)
    states = initial_states.copy()
    frac_infected = []

    for step in range(num_steps):
        new_states = states.copy()

        for node in range(N):
            neighbors = np.nonzero(G[node])[0]
            num_infected_neighbors = np.sum(states[neighbors])

            # Infection rule
            infection_prob = 1 - np.exp(-lambda_val * num_infected_neighbors)
            if states[node] == 0 and np.random.random() < infection_prob:
                new_states[node] = 1

            # Recovery rule
            recovery_prob = 1 - np.exp(-delta)
            if states[node] == 1 and np.random.random() < recovery_prob:
                new_states[node] = 0

        states = new_states.copy()
        frac_infected.append(np.sum(states) / N)

    return frac_infected


def main():
    N = 10**4  # Number of nodes
    k = 4      # Average degree
    p_values = np.logspace(-3, 0, 20)  # Range of rewiring probabilities (from 0.001 to 1)
    initial_fraction_infected = 0.01
    lambda_values = [0, 0.1, 0.3, 0.5, 1]
    delta = 1
    num_steps = 30

    # Measure clustering coefficient and degree distribution for different p values
    clustering_coefficients = []
    degree_distributions = []

    for p in p_values:
        G = watts_strogatz_model(N, k, p)

        # Measure clustering coefficient
        clustering_coefficient = measure_clustering_coefficient(G)
        clustering_coefficients.append(clustering_coefficient)

        # Measure degree distribution
        degree_probs = degree_distribution(G)
        degree_distributions.append(degree_probs)

    # Calculate the clustering coefficient ratio (C(p)/C(0))
    clustering_coefficients_normalized = [cc / clustering_coefficients[0] for cc in clustering_coefficients]

    # Plot the clustering coefficient as a function of p in log scale
    plt.plot(p_values, clustering_coefficients_normalized, marker='s', linestyle='-', color='b')
    plt.xscale('log')
    plt.xlabel('Rewiring Probability (p)')
    plt.ylabel('Clustering Coefficient C(p) / C(0)')
    plt.title('C(p) / C(0) as a Function of p in WS Model')
    plt.grid(True)
    plt.show()

    # Plot the SIS dynamics for different lambda values
    plt.figure(figsize=(10, 6))

    for lambda_val in lambda_values:
        G = watts_strogatz_model(N, k, p_values[0])  # Use the first value of p

        # Initialize the states of nodes
        initial_states = initialize_states(N, initial_fraction_infected)

        # Simulate the SIS dynamics
        frac_infected = sis_dynamics(G, initial_states, lambda_val, delta, num_steps)

        # Plot the fraction of infected nodes over time
        plt.plot(range(num_steps), frac_infected, label=f'λ = {lambda_val}')

    plt.xlabel('Time Step')
    plt.ylabel('Fraction of Infected Nodes')
    plt.title('SIS Dynamics on Watts-Strogatz Model')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
