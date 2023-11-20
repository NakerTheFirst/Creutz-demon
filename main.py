from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import random


class CreutzDemon:
    def __init__(self, matrix: np.ndarray[any], demons_num: int, init_demon_energies: list, iters_num: int):
        self.__matrix = matrix
        self.__matrix_size = np.size(self.__matrix[0])
        self.__demons_num = demons_num
        self.__init_demon_energies = init_demon_energies
        self.__demon_energy = 0
        self.__demon_energies = []
        self.__magnetisations = []
        self.__mean_magnetisations = []
        self.__temperatures = []
        self.__frequencies = []
        self.__stable_demon_energies = []
        self.__unique_stable_demon_energies = []
        self.__current_matrix_energy = 0
        self.__current_iter = 0
        self.__energy_delta = None
        self.__spin_coords = None
        self.__a = None
        self.__iters_num = list(range(1, iters_num + 1))
        self.__calculate_init_matrix_energy()

    def run(self):
        self.__init_lists()
        for i in range(self.__demons_num):
            self.__demon_energy = self.__init_demon_energies[i]

            for x in self.__iters_num:
                self.__randomise_spin_coords()
                self.__calculate_energy_delta()
                self.__handle_energy_exchange()
                self.__demon_energies[i].append(self.__demon_energy)
                self.__calculate_magnetisation()
            self.__save_stable_demon_energies()
            self.__normalise_magnetisation()
            self.__save_frequencies()
            self.__calculate_a()
            self.__calculate_temperature()
            self.__normalise_temperatures()
            self.__calculate_mean_magnetisation()
            self.__clear_attributes()
            self.__current_iter += 1

    def __init_lists(self):
        for x in range(self.__demons_num):
            self.__demon_energies.append([])
            self.__magnetisations.append([])
            self.__mean_magnetisations.append([])
            self.__temperatures.append([])
            self.__frequencies.append([])
            self.__stable_demon_energies.append([])
            self.__unique_stable_demon_energies.append([])

    def __calculate_init_matrix_energy(self):
        for i in range(self.__matrix_size):
            for j in range(self.__matrix_size):
                right_neighbor_j = (j + 1) % self.__matrix_size
                bottom_neighbor_i = (i + 1) % self.__matrix_size

                self.__current_matrix_energy -= self.__matrix[i][j] * self.__matrix[i][right_neighbor_j]
                self.__current_matrix_energy -= self.__matrix[i][j] * self.__matrix[bottom_neighbor_i][j]

    def __calculate_energy_delta(self):
        i, j = self.__spin_coords

        # Using modulo to implement Born-Karman boundary conditions
        top_neighbor_i = (i - 1) % self.__matrix_size
        bottom_neighbor_i = (i + 1) % self.__matrix_size
        left_neighbor_j = (j - 1) % self.__matrix_size
        right_neighbor_j = (j + 1) % self.__matrix_size

        sum_neighbors = (self.__matrix[top_neighbor_i][j] +
                         self.__matrix[bottom_neighbor_i][j] +
                         self.__matrix[i][left_neighbor_j] +
                         self.__matrix[i][right_neighbor_j])

        # Change in energy due to proposed spin flip
        self.__energy_delta = 2 * self.__matrix[i][j] * sum_neighbors

    def __handle_energy_exchange(self):
        if self.__energy_delta <= 0:
            self.__current_matrix_energy += self.__energy_delta
            self.__demon_energy -= self.__energy_delta
            self.__flip_spin()

        elif self.__demon_energy >= self.__energy_delta > 0:
            self.__current_matrix_energy += self.__energy_delta
            self.__demon_energy -= self.__energy_delta
            self.__flip_spin()

    def __normalise_magnetisation(self):
        self.__magnetisations[self.__current_iter] = [abs(m) for m in self.__magnetisations[self.__current_iter]]

        max_magnetisation = self.__matrix_size ** 2
        normalized_magnetisations = [m / max_magnetisation for m in self.__magnetisations[self.__current_iter]]

        # Overwrite the current magnetisations with the normalized values
        self.__magnetisations[self.__current_iter] = normalized_magnetisations

    def __save_stable_demon_energies(self):
        """Iterate over demon_energies, save all the energies after the first occurrence of 0"""
        flag = False
        for x in range(len(self.__demon_energies[self.__current_iter])):
            if self.__demon_energies[self.__current_iter][x] == 0 or flag or x > len(self.__iters_num) - 15:
                self.__stable_demon_energies[self.__current_iter].append(self.__demon_energies[self.__current_iter][x])
                flag = True

    def __calculate_a(self):
        log_y = np.log(self.__frequencies[self.__current_iter])
        x = np.array(self.__unique_stable_demon_energies[self.__current_iter])

        # Linear regression
        beta, alpha = np.polyfit(x, log_y, 1)

        if beta == 0:
            print("Beta is zero, this will lead to a temperature of infinity")
            pass

        self.__a = beta

    def __calculate_temperature(self):
        if self.__a == 0:
            print("a is zero, cannot calculate temperature")
            pass
        else:
            temperature = -1 / self.__a
            self.__temperatures[self.__current_iter] = temperature

    def __normalise_temperatures(self):
        # Example critical temperature for 2D Ising model
        T_c = 2.269
        self.__temperatures[self.__current_iter] = self.__temperatures[self.__current_iter] / T_c

    def __calculate_magnetisation(self):
        magnetisation = 0
        for i in range(self.__matrix_size):
            for j in range(self.__matrix_size):
                magnetisation += self.__matrix[i][j]
        self.__magnetisations[self.__current_iter].append(magnetisation)

    def __calculate_mean_magnetisation(self):
        interval = 2000
        iters_num = len(self.__iters_num)

        stabilised_magnetisation = []

        for j in range(iters_num - 1, iters_num - interval - 1, -1):
            stabilised_magnetisation.append(self.__magnetisations[self.__current_iter][j])

        mean_magnetisation = sum(stabilised_magnetisation) / len(stabilised_magnetisation)
        self.__mean_magnetisations[self.__current_iter].append(mean_magnetisation)

    def __flip_spin(self):
        self.__matrix[self.__spin_coords[0]][self.__spin_coords[1]] *= -1
        return self.__matrix

    def __save_frequencies(self):
        counter = Counter(self.__stable_demon_energies[self.__current_iter])
        self.__unique_stable_demon_energies[self.__current_iter] = list(counter.keys())
        self.__frequencies[self.__current_iter] = list(counter.values())

    def __randomise_spin_coords(self):
        self.__spin_coords = (random.randint(0, self.__matrix_size-1), random.randint(0, self.__matrix_size-1))

    def __clear_attributes(self):
        self.__current_matrix_energy = 0
        self.__energy_delta = None
        self.__spin_coords = None
        self.__a = None

    def plot_demon_energy_over_iterations(self, i):
        plt.figure(figsize=(10, 6))
        plt.plot(self.__iters_num, self.__demon_energies[i], '-o', markersize=3, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Demon Energy')
        plt.title('Demon Energy over Iterations')
        plt.grid(True)
        plt.show()

    def plot_demon_energy_histogram(self, i):
        # Linearise the frequencies using natural logarithm
        log_freq = np.log(self.__frequencies[i])

        plt.figure(figsize=(10, 6))
        plt.scatter(self.__unique_stable_demon_energies[i], log_freq, s=12)
        plt.xlabel("Energy")
        plt.ylabel("Log(frequency)")
        plt.title("Scatter Histogram of Demon Energy")
        plt.grid(True)
        plt.show()

    def plot_magnetisations_over_iterations(self, i):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.__iters_num, self.__magnetisations[i], 3)
        plt.title("Magnetisations over iterations")
        plt.xlabel('Iteration')
        plt.ylabel('Magnetisation')
        plt.grid(True)
        plt.show()

    def plot_magnetisation_to_temperature(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.__temperatures, self.__mean_magnetisations, 10)
        plt.title("Magnetisation to Temperature")
        plt.xlabel('Temperature')
        plt.ylabel('Magnetisation')
        plt.grid(True)
        plt.show()

    def get_mean_magnetisations(self):
        return self.__mean_magnetisations

    def get_temperatures(self):
        return self.__temperatures


def main():

    # Scrape the data
    with open("data.txt") as my_file:
        data = my_file.read()

    data_list = data.split()

    matrix_width = int(data_list[0])
    matrix_height = int(data_list[1])
    num_iterations = int(data_list[2])
    num_initial_demon_energies = int(data_list[3])
    demon_energy_samples = [int(energy) for energy in data_list[4:4 + num_initial_demon_energies]]

    matrix = np.empty(shape=(matrix_width, matrix_height))
    matrix.fill(1)

    demon = CreutzDemon(matrix, num_initial_demon_energies, demon_energy_samples, num_iterations)
    demon.run()

    demon.plot_demon_energy_over_iterations(6)
    demon.plot_demon_energy_histogram(6)

    demon.plot_magnetisations_over_iterations(6)
    demon.plot_magnetisation_to_temperature()

    return 0


if __name__ == "__main__":
    main()
