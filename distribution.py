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


# Define the degree distribution function
def degree_distribution(G):
    N = len(G)
    degrees = np.sum(G, axis=1)
    degree_counts = np.bincount(degrees)
    degree_probs = degree_counts / N
    return degree_probs


# Main function definition to execute
def main():
    N = 10**4  # Number of nodes
    k = 4      # Average degree
    p_values = [0, 0.5, 1]

    # Measure the clustering coefficient and degree distribution for different p values
    clustering_coefficients = []
    degree_distributions = []

    for p in p_values:
        G = watts_strogatz_model(N, k, p)

        # Measure the clustering coefficient
        clustering_coefficient = measure_clustering_coefficient(G)
        clustering_coefficients.append(clustering_coefficient)

        # Measure the degree distribution
        degree_probs = degree_distribution(G)
        degree_distributions.append(degree_probs)

    # Plot the degree distribution for different p values in three subplots (bar plots)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for i, p in enumerate(p_values):
        axs[i].bar(range(len(degree_distributions[i])), degree_distributions[i], align='center')
        axs[i].set_title(f'Rewiring Probability (p) = {p:.1f}')
        axs[i].set_xlim(0, 11)
        axs[i].set_ylim(0, 1.1)
        axs[i].set_xlabel('Degree k')
        axs[i].set_ylabel('Probability p(k)')
        axs[i].grid(True)

    plt.suptitle('p(k) for Different p in WS Model')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
