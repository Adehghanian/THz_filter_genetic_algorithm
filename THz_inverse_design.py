import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
import logging
import random
import matplotlib.colors as mcolors
import multiprocessing as mp
from itertools import product

# Setup logging for debugging and information purposes
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

class GridOptimizer:
    def __init__(self, rows, cols, f_center, bandwidth, frequencies):
        """
        Initialize the grid optimizer with design parameters.
        
        Parameters:
        - rows: Number of rows in the grid.
        - cols: Number of columns in the grid.
        - f_center: Center frequency for optimization.
        - bandwidth: Bandwidth for the filter.
        - frequencies: Array of frequencies to analyze.
        """
        self.rows = rows
        self.cols = cols
        self.bandwidth = bandwidth
        self.frequencies = frequencies
        self.Eeff = 1.18  # Effective permittivity
        self.c = 3e8  # Speed of light (m/s)
        # Calculate cell size based on frequency and permittivity
        self.cell_size = self.c / (2 * f_center * np.sqrt(self.Eeff))
        self.best_grid = None  # Best grid configuration
        self.best_reward = float('-inf')  # Best reward score
        self.best_S21_dB = None  # Best S21 parameter (dB)
        self.best_S11_dB = None  # Best S11 parameter (dB)
        # Fixed column configuration for testing purposes
        self.fixed_column = np.array([0, 0, 0, 1, 0, 0, 0])
        # Generate ideal band-stop filter parameters
        self.ideal_S11, self.ideal_S21, self.ideal_S11_phase, self.ideal_S21_phase = self.ideal_band_stop_filter()

        # Generate valid column configurations based on constraints
        possible_columns = list(product([0, 1], repeat=(rows-1)//2))
        valid_columns = [np.array(col) for col in possible_columns if np.sum(np.array(col) ^ np.append(np.array(col)[1:], 1)) < 2]
        valid_columns = np.array(valid_columns).T

        # Combine rows to form complete columns
        middle_row = np.ones((1, valid_columns.shape[1]))  # Middle row
        mirrored_col = np.flipud(valid_columns)  # Mirror valid columns
        combined_grid = np.vstack((valid_columns, middle_row, mirrored_col))
        self.valid_columns = [combined_grid[:, i] for i in range(combined_grid.shape[1])]

    def calculate_S_W_values(self, grid):
        """
        Calculate strip width (W) and spacing (S) values for the grid.
        """
        W_values = np.sum(grid, axis=0) * 5e-6 + 40e-6
        S_values = 115e-6 - W_values
        return S_values, W_values

    def calculate_Z1(self, S_values, W_values):
        """
        Calculate characteristic impedance (Z1) values based on grid parameters.
        """
        h = 1e-6  # Thickness of substrate
        K_values = S_values / (S_values + 2 * W_values)
        KP_values = np.sqrt(1 - K_values**2)
        K1_values = (np.sinh(np.pi * S_values / (4 * h))) / (np.sinh(np.pi * (S_values + 2 * W_values) / (4 * h)))
        K1p_values = np.sqrt(1 - K1_values**2)
        eps_eff = 1 + ((1.3 - 1) / 2) * (ellipk(KP_values) / ellipk(K_values)) * (ellipk(K1_values) / ellipk(K1p_values))
        Z1_values = 120 * np.pi * ellipk(K_values) / (np.sqrt(eps_eff) * ellipk(KP_values))
        return Z1_values

    def ideal_band_stop_filter(self):
        """
        Generate an ideal band-stop filter grid and calculate S-parameters.
        """
        first_column = np.array([[0], [0], [0], [1], [0], [0], [0]])  # Fixed column
        pattern = np.array([[1, 0], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0]])
        ideal_grid = np.hstack((first_column, np.tile(pattern, (1, (self.cols+1)//2))))
        S, W_values = self.calculate_S_W_values(ideal_grid)
        Z1_values = self.calculate_Z1(S, W_values)
        ideal_S11, ideal_S21, S11_phases, S21_phases = self.calculate_S21_dB(Z1_values)
        return ideal_S11, ideal_S21, S11_phases, S21_phases

    def calculate_S21_dB(self, Z1_values):
        """
        Calculate S21 and S11 parameters in dB and their phases.
        """
        S21_values_dB = []
        S11_values_dB = []
        S21_phases = []
        S11_phases = []

        for F in self.frequencies:
            beta = (2 * np.pi * F * np.sqrt(self.Eeff)) / self.c  # Phase constant
            TL = np.eye(2, dtype=complex)  # Transmission line matrix

            for i in range(self.cols+2):
                Z1 = Z1_values[i]
                T1 = self.calculate_matrix(beta, self.cell_size/2, Z1)
                TL = np.dot(T1, TL)

            S11, S21 = self.calculate_scattering_parameters(TL, Z1_values[0] if Z1_values[0] != 0 else 1)

            S21_dB = 20 * np.log10(np.abs(S21))
            S11_dB = 20 * np.log10(np.abs(S11))
            S21_phase = np.angle(S21, deg=True)
            S11_phase = np.angle(S11, deg=True)

            S21_values_dB.append(S21_dB)
            S11_values_dB.append(S11_dB)
            S21_phases.append(S21_phase)
            S11_phases.append(S11_phase)

        return np.array(S11_values_dB), np.array(S21_values_dB), np.array(S11_phases), np.array(S21_phases)

    def calculate_matrix(self, beta, L, Z1):
        """
        Calculate the transmission line matrix for a given section.
        """
        A = np.cos(beta * L)
        B = Z1 * 1j * np.sin(beta * L)
        C = 1j * np.sin(beta * L) / Z1 if Z1 != 0 else 0
        D = np.cos(beta * L)
        return np.array([[A, B], [C, D]])

    def calculate_scattering_parameters(self, TL, Z1):
        """
        Calculate S11 and S21 scattering parameters from transmission matrix.
        """
        numerator_S11 = TL[0, 0] + TL[0, 1] / Z1 - TL[1, 0] * Z1 - TL[1, 1]
        denominator_S11_S21 = TL[0, 0] + TL[0, 1] / Z1 + TL[1, 0] * Z1 + TL[1, 1]

        S11 = numerator_S11 / denominator_S11_S21
        S21 = 2 / denominator_S11_S21

        return S11, S21

    def calculate_reward(self, grid):
        """
        Calculate the reward for a grid configuration based on deviation from the ideal response.
        """
        try:
            S, W_values = self.calculate_S_W_values(grid)
            Z1_values = self.calculate_Z1(S, W_values)
            S11_dB, S21_dB, S11_phases, S21_phases = self.calculate_S21_dB(Z1_values)
            rmse = 0.4 * np.sqrt(np.mean((S21_dB - self.ideal_S21) ** 2)) + \
                   0.4 * np.sqrt(np.mean((S11_dB - self.ideal_S11) ** 2)) + \
                   0.1 * np.sqrt(np.mean((S11_phases - self.ideal_S11_phase) ** 2)) + \
                   0.1 * np.sqrt(np.mean((S21_phases - self.ideal_S21_phase) ** 2))
            return -rmse, S11_dB, S21_dB, S11_phases, S21_phases
        except Exception as e:
            logging.error(f"Error calculating reward: {e}")
            return float('-inf'), None, None, None, None

    def visualize_best_grid(self, base_filename='Combined_plot'):
        if self.best_grid is not None:
            # Mirror the grid and add a gray middle row
            mirrored_grid = np.flipud(self.best_grid)
            combined_grid = np.insert(np.vstack((mirrored_grid, self.best_grid)), len(mirrored_grid), 0, axis=0)

            gray_middle_row = np.zeros((1, combined_grid.shape[1]))
            middle_index = len(combined_grid) // 2
            combined_grid = np.vstack((combined_grid[:middle_index], gray_middle_row, combined_grid[middle_index:]))

            # Add gray padding rows
            gray_row = np.zeros((4, combined_grid.shape[1]))
            combined_grid = np.vstack((gray_row, combined_grid, gray_row))
            # Define the color map for the grid
            cmap = mcolors.ListedColormap(['gray', 'gold'])
            bounds = [-0.5, 0.5, 1.5]
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Calculate S-parameters
            _, S11, S21, S11_phase, S21_phase = self.calculate_reward(self.best_grid)

            # Create a figure with 3 subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.5, 1, 1]})

            # Plot the grid visualization
            axs[0].imshow(combined_grid, cmap=cmap, norm=norm, interpolation='nearest')
            axs[0].axis('off')
            axs[0].set_title(f'Grid Visualization\nReward: {self.best_reward:.2f}, Generation: {self.visualize_gen}', fontsize=14)

            # Plot S11 and S21 Magnitudes
            axs[1].plot(self.frequencies, S21, label='Calculated S21 (dB)', color='blue')
            axs[1].plot(self.frequencies, self.ideal_S21, label='Ideal S21 (dB)', linestyle='--', color='cyan')
            axs[1].plot(self.frequencies, S11, label='Calculated S11 (dB)', color='red')
            axs[1].plot(self.frequencies, self.ideal_S11, label='Ideal S11 (dB)', linestyle='--', color='orange')
            axs[1].set_xlabel('Frequency (Hz)')
            axs[1].set_ylabel('Magnitude (dB)')
            axs[1].set_title('S11 and S21 Magnitude Comparison')
            axs[1].legend()
            axs[1].grid(True)

            # Plot S11 and S21 Phases
            axs[2].plot(self.frequencies, S21_phase, label='Calculated S21 Phase (deg)', color='blue')
            axs[2].plot(self.frequencies, self.ideal_S21_phase, label='Ideal S21 Phase (deg)', linestyle='--', color='cyan')
            axs[2].plot(self.frequencies, S11_phase, label='Calculated S11 Phase (deg)', color='red')
            axs[2].plot(self.frequencies, self.ideal_S11_phase, label='Ideal S11 Phase (deg)', linestyle='--', color='orange')
            axs[2].set_xlabel('Frequency (Hz)')
            axs[2].set_ylabel('Phase (Degrees)')
            axs[2].set_title('S11 and S21 Phase Comparison')
            axs[2].legend()
            axs[2].grid(True)

            # Adjust layout and save the plot
            plt.tight_layout(pad=3.0)
            filename = f"{base_filename}.png"
            plt.savefig(filename, dpi=300)
            print(f"Combined plot saved as {filename}")
            plt.show()
        else:
            logging.warning("No valid grid or reward available.")



class GeneticAlgorithmOptimizer(GridOptimizer):
    def __init__(self, rows, cols, f_center, bandwidth, frequencies, population_size=1000, generations=500, mutation_rate=0.3, max_mutation_attempts=500):

        super().__init__(rows, cols, f_center, bandwidth, frequencies)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.visualize_gen = 0
        self.max_mutation_attempts = max_mutation_attempts


    def initialize_population(self):
        # Ensure valid_columns is a NumPy array
        valid_columns_array = np.array(self.valid_columns)

        # Initialize the population with a structured random selection from valid columns
        population = []
        num_valid_columns = valid_columns_array.shape[0]
        
        while len(population) < self.population_size:
            grid = []
            for j in range(self.cols):
                
                if np.random.rand() < 0.99:  
                    # Almost all the grids are chosen randomly. In the case of a periodic filter design, 
                    # we can adjust this value. This helps the process reach the optimized periodic grid faster.
                    column = valid_columns_array[np.random.choice(num_valid_columns)]
                else:
                    column = valid_columns_array[-1] if j % 2 == 0 else valid_columns_array[0]
                grid.append(column)
            grid = np.vstack((self.fixed_column, grid, self.fixed_column))
            population.append(np.array(grid).T)
        
        return population

    def select_parents(self, population, rewards):    ####roulette-wheel selection
        # Selects two parents based on rewards
        total_fitness = sum(rewards)
        selection_probs = [r / total_fitness for r in rewards]
        return random.choices(population, weights=selection_probs, k=2)
    
    def select_parents_tournament(self, population, rewards, tournament_size=100):
        parents = []
        for _ in range(2):  # Select two parents
            tournament = random.sample(list(zip(population, rewards)), tournament_size)
            best_individual = max(tournament, key=lambda x: x[1])[0]
            parents.append(best_individual)
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
            # Generate two crossover points
            crossover_point1 = np.random.randint(2, self.cols - 1)
            crossover_point2 = np.random.randint(crossover_point1 + 1, self.cols)
            
            # Create children by swapping the segments between the two crossover points
            child1 = np.hstack((parent1[:, :crossover_point1], 
                                parent2[:, crossover_point1:crossover_point2], 
                                parent1[:, crossover_point2:]))
            child2 = np.hstack((parent2[:, :crossover_point1], 
                                parent1[:, crossover_point1:crossover_point2], 
                                parent2[:, crossover_point2:]))
            
            return child1, child2

    def mutate(self, grid, current_generation):
        # Calculate effective mutation rate once
        effective_mutation_rate = self.mutation_rate * (current_generation / self.generations)
        
        # Generate a mask for columns (1-based indexing)
        mutation_mask = np.random.rand(self.cols) < effective_mutation_rate

        # Perform mutation on columns selected by the mask
        for j in np.where(mutation_mask)[0] + 1:  # Adjust to 1-based index
            # Select a random column from valid_columns
            new_column = self.valid_columns[np.random.randint(len(self.valid_columns))]
            # Apply the new column to the grid
            grid[:, j] = new_column.reshape(-1)  # Adjust back to 0-based index for assignment

        return grid


    def genetic_algorithm_search(self):
        # Initialize population and pool for multiprocessing
        population = self.initialize_population()
        pool = mp.Pool(mp.cpu_count())  # Use all available CPUs for parallelization

        for generation in range(self.generations):
            # Calculate rewards in parallel
            rewards = pool.map(self.calculate_reward_parallel, population)

            # Get the best reward and corresponding grid
            max_reward, best_S11_dB, best_S21_dB, best_S11_phase, best_S21_phase = max(rewards, key=lambda x: x[0])

            if max_reward > self.best_reward:
                self.best_reward = max_reward
                best_index = rewards.index((max_reward, best_S11_dB, best_S21_dB, best_S11_phase, best_S21_phase))
                self.best_grid = population[best_index]
                self.visualize_gen = generation
                #self.visualize_best_grid()

            logging.info(f"Generation {generation + 1}, Best Reward: {self.best_reward:.2f}")
            if self.best_reward == 0:
                break
            # Create the next generation through crossover and mutation
            new_population = []
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents_tournament(population, [r[0] for r in rewards])
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, generation)
                child2 = self.mutate(child2, generation)
                new_population.append(child1)
                new_population.append(child2)

            population = new_population[:self.population_size]

        logging.info(f"Best Reward after {self.generations} generations: {self.best_reward:.2f}")
        pool.close()  # Close the pool
        pool.join()  # Wait for the worker processes to terminate
    
    def calculate_reward_parallel(self, grid):
        reward, S11_dB, S21_dB, S11_phase, S21_phase = self.calculate_reward(grid)
        return reward, S11_dB, S21_dB, S11_phase, S21_phase
    def printval(self):
        print(f"Population Size: {self.population_size}, Generations: {self.generations}, "
          f"Mutation Rate: {self.mutation_rate}, Max Mutation Attempts: {self.max_mutation_attempts}")



if __name__ == '__main__':
    rows, cols = 7, 41
    c = 3e8
    f_center = 0.8e12
    bandwidth = 0.4e12
    frequencies = np.arange(4e11, 1.2e12, 5e9)

    # Initialize Genetic Algorithm Optimizer
    ga_optimizer = GeneticAlgorithmOptimizer(rows, cols, f_center, bandwidth, frequencies)
    ga_optimizer.genetic_algorithm_search()
    ga_optimizer.visualize_best_grid()
    #ga_optimizer.render_S21()
    ga_optimizer.printval()
    