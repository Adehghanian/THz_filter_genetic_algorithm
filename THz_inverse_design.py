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
    """
    A class to optimize a grid for a filter design based on electromagnetic properties and frequency parameters.

        Attributes:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        frequency_center (float): Center frequency for optimization in Hz.
        bandwidth (float): Bandwidth for the filter in Hz.
        frequencies (np.ndarray): Array of frequencies to analyze.
        cell_size (float): Calculated cell size based on center frequency and effective permittivity.
        best_grid (np.ndarray): Best grid configuration found during optimization.
        best_reward (float): Best reward score obtained during optimization.
        best_S21_dB (float): Best S21 parameter (dB) found.
        best_S11_dB (float): Best S11 parameter (dB) found.
        fixed_column (np.ndarray): Fixed column configuration for testing purposes.
        valid_columns (List[np.ndarray]): List of valid column configurations that satisfy constraints.

        Methods:
        __init__(self, rows: int, cols: int, frequency_center: float, bandwidth: float, frequencies: np.ndarray):
        Initializes the grid optimizer with design parameters.
    """
        
    def __init__(self, rows: int, cols: int, frequency_center: float, bandwidth: float, frequencies: np.ndarray):

        """
        Initializes the grid optimizer with design parameters.

        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            frequency_center (float): Center frequency for optimization in Hz.
            bandwidth (float): Bandwidth for the filter in Hz.
            frequencies (np.ndarray): Array of frequencies to analyze.

        Raises:
            ValueError: If center frequency is zero.
        """
        # Set design parameters
        self.rows = rows
        self.cols = cols
        self.bandwidth = bandwidth
        self.frequencies = frequencies
        
        # Constants
        self.effective_permittivity = 1.18  # Effective permittivity
        self.speed_of_light = 3e8  # Speed of light in m/s

        # Validate center frequency
        if frequency_center == 0:
            raise ValueError("Center frequency (frequency_center) cannot be zero.")
        
        # Calculate cell size based on frequency and permittivity
        self.cell_size = self.speed_of_light / (2 * frequency_center * np.sqrt(self.effective_permittivity))
        
        # Initialize optimization results
        self.best_grid = None  # Best grid configuration
        self.best_reward = float('-inf')  # Best reward score
        self.best_S21_dB = None  # Best S21 parameter (dB)
        self.best_S11_dB = None  # Best S11 parameter (dB)
        
        # Fixed column configuration for testing purposes
        self.fixed_column = np.array([0, 0, 0, 1, 0, 0, 0])

        # Generate ideal band-stop filter parameters
        self.ideal_S11, self.ideal_S21, self.ideal_S11_phase, self.ideal_S21_phase = self.ideal_band_stop_filter()

        # Generate valid column configurations based on constraints
        possible_columns = list(product([0, 1], repeat=(self.rows - 1) // 2))
        valid_columns = [np.array(col) for col in possible_columns if np.sum(np.array(col) ^ np.append(np.array(col)[1:], 1)) < 2]
        valid_columns = np.array(valid_columns).T

        # Combine rows to form complete columns
        middle_row = np.ones((1, valid_columns.shape[1]))  # Middle row
        mirrored_col = np.flipud(valid_columns)  # Mirror valid columns
        combined_grid = np.vstack((valid_columns, middle_row, mirrored_col))
        
        # List of valid columns for optimization
        self.valid_columns = [combined_grid[:, i] for i in range(combined_grid.shape[1])]

    def calculate_S_W_values(self, grid: np.ndarray) -> np.ndarray:
        """
        Calculate strip width (W) and spacing (S) values for the grid.
        """
        W_values = np.sum(grid, axis=0) * 5e-6 + 40e-6
        S_values = 115e-6 - W_values
        return S_values, W_values

    def calculate_Z1(self, S_values: np.ndarray, W_values: np.ndarray) -> np.ndarray:
        """
        Calculate the characteristic impedance (Z1) values based on the grid parameters.

        This method computes the characteristic impedance (Z1) values based on the input S-parameters and
        W-values, utilizing elliptic integrals to adjust for the effective permittivity and substrate thickness.

        Args:
            S_values (np.ndarray): Array of S-parameters (typically S11 or S21).
            W_values (np.ndarray): Array of corresponding W-values related to the grid.

        Returns:
            np.ndarray: Calculated Z1 values for each input pair of S_values and W_values.
        """
        # Substrate thickness (in meters)
        h = 1e-6  

        # Calculate K and KP values using the S and W values
        K_values = S_values / (S_values + 2 * W_values)
        KP_values = np.sqrt(1 - K_values**2)

        # Calculate K1 and K1' values based on the S and W values
        K1_values = (np.sinh(np.pi * S_values / (4 * h))) / (np.sinh(np.pi * (S_values + 2 * W_values) / (4 * h)))
        K1p_values = np.sqrt(1 - K1_values**2)

        # Compute the effective permittivity (eps_eff) using elliptic integrals
        eps_eff = 1 + ((1.3 - 1) / 2) * (ellipk(KP_values) / ellipk(K_values)) * (ellipk(K1_values) / ellipk(K1p_values))

        # Calculate the characteristic impedance (Z1) values using the effective permittivity
        Z1_values = 120 * np.pi * ellipk(K_values) / (np.sqrt(eps_eff) * ellipk(KP_values))

        return Z1_values

    def ideal_band_stop_filter(self):
        """
        Generate an ideal band-stop filter grid and calculate S-parameters.

        This method constructs a band-stop filter grid using a fixed first column 
        and a repeating pattern for the remaining columns. It then calculates 
        the S-parameters for the grid.

        Returns:
            tuple: Ideal S11, S21 parameters (dB), and their phases.
        """
        # Define the fixed first column for the grid
        first_column = np.array([[0], [0], [0], [1], [0], [0], [0]])  # Fixed column
        
        # Define the repeating pattern for the rest of the grid
        pattern = np.array([[1, 0], [1, 0], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0]])
        
        # Create the full grid by concatenating the first column and the repeating pattern
        ideal_grid = np.hstack((first_column, np.tile(pattern, (1, (self.cols+1)//2))))
        
        # Calculate S and W values using the ideal grid
        S, W_values = self.calculate_S_W_values(ideal_grid)
        
        # Calculate Z1 values based on S and W values
        Z1_values = self.calculate_Z1(S, W_values)
        
        # Calculate S-parameters (S11, S21) and their phases from Z1 values
        ideal_S11, ideal_S21, S11_phases, S21_phases = self.calculate_S21_dB(Z1_values)
        
        # Return the computed S-parameters and phases
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
            beta = (2 * np.pi * F * np.sqrt(self.effective_permittivity)) / self.speed_of_light  # Phase constant
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

    def calculate_matrix(self, beta: float, L: float, Z1: float) -> np.ndarray:
        """
        Calculate the transmission line matrix for a given section.

        This method computes the transmission line matrix, which is used to model 
        the behavior of the transmission line for a specific section based on 
        the propagation constant (beta), length (L), and characteristic impedance (Z1).

        Args:
            beta (float): The propagation constant of the transmission line (in radians per unit length).
            L (float): The length of the transmission line section (in meters).
            Z1 (float): The characteristic impedance of the transmission line (in ohms).

        Returns:
            np.ndarray: The 2x2 transmission line matrix representing the section.
        """
        # Calculate elements of the transmission line matrix
        A = np.cos(beta * L)  # Element A (cosine of beta * L)
        B = Z1 * 1j * np.sin(beta * L)  # Element B (Z1 * sine of beta * L)
        
        # Element C (sine of beta * L divided by Z1), handled if Z1 is zero to avoid division by zero
        C = 1j * np.sin(beta * L) / Z1 if Z1 != 0 else 0
        
        # Element D (cosine of beta * L)
        D = np.cos(beta * L)

        # Return the transmission line matrix as a 2x2 numpy array
        return np.array([[A, B], [C, D]])

    def calculate_scattering_parameters(self, TL: np.ndarray, Z1: float):
        """
        Calculate the S11 and S21 scattering parameters from the transmission matrix.

        This method computes the reflection (S11) and transmission (S21) scattering 
        parameters for a given transmission line (TL) matrix and characteristic impedance (Z1).

        Args:
            TL (np.ndarray): The 2x2 transmission matrix representing the section.
            Z1 (float): The characteristic impedance of the transmission line (in ohms).

        Returns:
            tuple: S11 and S21 scattering parameters.
        """
        # Calculate the numerator and denominator for S11 and S21
        numerator_S11 = TL[0, 0] + TL[0, 1] / Z1 - TL[1, 0] * Z1 - TL[1, 1]
        denominator_S11_S21 = TL[0, 0] + TL[0, 1] / Z1 + TL[1, 0] * Z1 + TL[1, 1]

        # Calculate S11 and S21 parameters
        S11 = numerator_S11 / denominator_S11_S21
        S21 = 2 / denominator_S11_S21

        return S11, S21


    def calculate_reward(self, grid: np.ndarray):
        """
        Calculate the reward for a given grid configuration based on deviation from the ideal response.

        This method computes the root mean square error (RMSE) between the calculated and ideal 
        S-parameters (S11, S21) and their phases. The reward is negative RMSE, where a lower RMSE 
        corresponds to a higher reward.

        Args:
            grid (np.ndarray): The current grid configuration to evaluate.

        Returns:
            tuple: A tuple containing:
                - reward (float): The calculated reward (negative RMSE).
                - S11_dB (np.ndarray): Calculated S11 parameters in dB.
                - S21_dB (np.ndarray): Calculated S21 parameters in dB.
                - S11_phases (np.ndarray): Calculated S11 phases.
                - S21_phases (np.ndarray): Calculated S21 phases.
        """
        try:
            # Calculate S, W values, and Z1 from the grid
            S, W_values = self.calculate_S_W_values(grid)
            Z1_values = self.calculate_Z1(S, W_values)
            
            # Calculate S-parameters (S11, S21) and their phases
            S11_dB, S21_dB, S11_phases, S21_phases = self.calculate_S21_dB(Z1_values)
            
            # Compute RMSE as the reward metric (lower RMSE is better)
            rmse = 0.4 * np.sqrt(np.mean((S21_dB - self.ideal_S21) ** 2)) + \
                0.4 * np.sqrt(np.mean((S11_dB - self.ideal_S11) ** 2)) + \
                0.1 * np.sqrt(np.mean((S11_phases - self.ideal_S11_phase) ** 2)) + \
                0.1 * np.sqrt(np.mean((S21_phases - self.ideal_S21_phase) ** 2))

            # Return negative RMSE and S-parameters/phases
            return -rmse, S11_dB, S21_dB, S11_phases, S21_phases
        
        except Exception as e:
            # Log error if reward calculation fails
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
    def __init__(self, 
                    rows: int, 
                    cols: int, 
                    f_center: float, 
                    bandwidth: float, 
                    frequencies: np.ndarray, 
                    population_size: int = 1000, 
                    generations: int = 500, 
                    mutation_rate: float = 0.3, 
                    max_mutation_attempts: int = 500
                    ):
            """
            Initialize the genetic algorithm optimizer with design parameters.

            Args:
                rows (int): Number of rows in the grid.
                cols (int): Number of columns in the grid.
                f_center (float): Center frequency for optimization in Hz.
                bandwidth (float): Bandwidth for the filter in Hz.
                frequencies (np.ndarray): Array of frequencies to analyze.
                population_size (int, optional): The size of the population for the genetic algorithm. Defaults to 1000.
                generations (int, optional): The number of generations to run the genetic algorithm. Defaults to 500.
                mutation_rate (float, optional): The mutation rate for the genetic algorithm. Defaults to 0.3.
                max_mutation_attempts (int, optional): The maximum number of mutation attempts. Defaults to 500.
            """
            super().__init__(rows, cols, f_center, bandwidth, frequencies)  # Initialize parent class
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
            self.visualize_gen = 0
            self.max_mutation_attempts = max_mutation_attempts


    def initialize_population(self):
        """
        Initialize the population for the genetic algorithm with a random selection of valid columns.

        This method creates an initial population of grid configurations by randomly selecting columns 
        from the list of valid columns. The majority of the grid columns are selected randomly, 
        while a small portion may use fixed values for periodic grid configurations.

        Returns:
            list: A list of grid configurations, each represented as a numpy array.
        """
        # Ensure valid_columns is a NumPy array
        valid_columns_array = np.array(self.valid_columns)

        # Initialize the population with a structured random selection from valid columns
        population = []
        num_valid_columns = valid_columns_array.shape[0]
        
        while len(population) < self.population_size:
            grid = []
            for j in range(self.cols):
                
                if np.random.rand() < 0.99:  
                    # Most grids are chosen randomly. Adjust this value for specific filter designs.
                    column = valid_columns_array[np.random.choice(num_valid_columns)]
                else:
                    # Occasionally, use fixed columns for periodic design symmetry.
                    column = valid_columns_array[-1] if j % 2 == 0 else valid_columns_array[0]
                grid.append(column)
            
            # Stack fixed columns at top and bottom for boundary conditions
            grid = np.vstack((self.fixed_column, grid, self.fixed_column))
            population.append(np.array(grid).T)
        
        return population

    def select_parents(self, population, rewards):
        """
        Select two parents using roulette-wheel selection based on fitness (rewards).

        In roulette-wheel selection, individuals are selected based on their relative fitness. 
        The higher the fitness, the greater the chance of being selected.

        Args:
            population (list): The list of potential parent grid configurations.
            rewards (list): The list of fitness scores (rewards) for each individual in the population.

        Returns:
            tuple: Two selected parents (grid configurations).
        """
        # Calculate total fitness
        total_fitness = sum(rewards)
        
        # Calculate selection probabilities based on fitness
        selection_probs = [r / total_fitness for r in rewards]
        
        # Select two parents based on the calculated probabilities
        return random.choices(population, weights=selection_probs, k=2)
    
    def select_parents_tournament(self, population, rewards, tournament_size=100):
        """
        Select two parents using tournament selection.

        In tournament selection, a subset of the population is randomly selected, 
        and the individual with the highest fitness in the subset is chosen as a parent. 
        This process is repeated to select two parents.

        Args:
            population (list): The list of potential parent grid configurations.
            rewards (list): The list of fitness scores (rewards) for each individual in the population.
            tournament_size (int): The number of individuals randomly selected for each tournament.

        Returns:
            tuple: Two selected parents (grid configurations).
        """
        parents = []
        for _ in range(2):  # Select two parents
            # Randomly sample the population for the tournament
            tournament = random.sample(list(zip(population, rewards)), tournament_size)
            
            # Select the individual with the highest fitness from the tournament
            best_individual = max(tournament, key=lambda x: x[1])[0]
            parents.append(best_individual)
        
        return parents[0], parents[1]

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce two offspring.

        The crossover operation exchanges parts of the parents' grids to create new children.
        Two crossover points are selected randomly, and the portions between those points are swapped.

        Args:
            parent1 (np.ndarray): The first parent grid configuration.
            parent2 (np.ndarray): The second parent grid configuration.

        Returns:
            tuple: Two child grid configurations created through crossover.
        """
        # Generate two crossover points
        crossover_point1 = np.random.randint(2, self.cols - 1)
        crossover_point2 = np.random.randint(crossover_point1 + 1, self.cols)
        
        # Swap segments between the two parents to create the children
        child1 = np.hstack((parent1[:, :crossover_point1], 
                            parent2[:, crossover_point1:crossover_point2], 
                            parent1[:, crossover_point2:]))
        child2 = np.hstack((parent2[:, :crossover_point1], 
                            parent1[:, crossover_point1:crossover_point2], 
                            parent2[:, crossover_point2:]))
        
        return child1, child2


    def mutate(self, grid, current_generation):
        """
        Apply mutation to the grid with a generation-dependent mutation rate.

        This method applies mutation to the grid by replacing certain columns 
        with randomly selected columns from the list of valid columns. The 
        mutation rate increases as the generation progresses, allowing for 
        more exploration early on and focusing on exploitation in later generations.

        Args:
            grid (np.ndarray): The current grid configuration to mutate.
            current_generation (int): The current generation number in the evolutionary process.

        Returns:
            np.ndarray: The mutated grid configuration.
        """
        # Calculate the effective mutation rate for the current generation
        effective_mutation_rate = self.mutation_rate * (current_generation / self.generations)
        
        # Generate a mutation mask, determining which columns will be mutated
        mutation_mask = np.random.rand(self.cols) < effective_mutation_rate

        # Perform mutation on the columns selected by the mutation mask
        for j in np.where(mutation_mask)[0]:  # Indices are 0-based by default
            # Select a random column from valid_columns
            new_column = self.valid_columns[np.random.randint(len(self.valid_columns))]
            # Apply the new column to the grid at the selected column index
            grid[:, j] = new_column.reshape(-1)  # Ensure the column fits into the grid shape

        return grid


    def genetic_algorithm_search(self):
        """
        Perform the genetic algorithm search to optimize grid configurations.
        
        This method initializes the population, performs parallelized reward calculations, 
        and iterates through generations, selecting parents, performing crossover, 
        and applying mutation to evolve the population towards the optimal solution.
        """
        # Initialize population and pool for multiprocessing
        population = self.initialize_population()
        pool = mp.Pool(mp.cpu_count())  # Use all available CPUs for parallelization

        for generation in range(self.generations):
            # Calculate rewards in parallel
            rewards = pool.map(self.calculate_reward_parallel, population)

            # Get the best reward and corresponding grid
            max_reward, best_S11_dB, best_S21_dB, best_S11_phase, best_S21_phase = max(rewards, key=lambda x: x[0])

            # Update the best reward and grid if a better one is found
            if max_reward > self.best_reward:
                self.best_reward = max_reward
                best_index = rewards.index((max_reward, best_S11_dB, best_S21_dB, best_S11_phase, best_S21_phase))
                self.best_grid = population[best_index]
                self.visualize_gen = generation
                # self.visualize_best_grid()  # Optional: visualize the best grid at each generation

            # Log progress of the genetic algorithm
            logging.info(f"Generation {generation + 1}, Best Reward: {self.best_reward:.2f}")
            if self.best_reward == 0:
                logging.info(f"Optimal solution reached in Generation {generation + 1}.")
                break

            # Create the next generation through crossover and mutation
            population = self.create_next_generation(population, rewards, generation)

        logging.info(f"Best Reward after {self.generations} generations: {self.best_reward:.2f}")
        pool.close()  # Close the pool
        pool.join()  # Wait for the worker processes to terminate

    def create_next_generation(self, population, rewards, generation):
        """
        Create the next generation of the population through crossover and mutation.

        Args:
            population (list): The current population of grid configurations.
            rewards (list): The rewards corresponding to each individual in the population.
            generation (int): The current generation number to adjust mutation rate.

        Returns:
            list: The new generation of grid configurations after crossover and mutation.
        """
        new_population = []
        while len(new_population) < self.population_size:
            # Select two parents based on tournament selection
            parent1, parent2 = self.select_parents_tournament(population, [r[0] for r in rewards])
            # Perform crossover to create two children
            child1, child2 = self.crossover(parent1, parent2)
            # Apply mutation to the children
            child1 = self.mutate(child1, generation)
            child2 = self.mutate(child2, generation)
            new_population.append(child1)
            new_population.append(child2)

        # Ensure the new population size does not exceed the desired population size
        return new_population[:self.population_size]

    def calculate_reward_parallel(self, grid):
        """
        Wrapper for reward calculation to be used in parallel processing.

        Args:
            grid (np.ndarray): The grid configuration for which the reward is to be calculated.

        Returns:
            tuple: Reward and S-parameters (S11, S21) in dB and their phases.
        """
        reward, S11_dB, S21_dB, S11_phase, S21_phase = self.calculate_reward(grid)
        return reward, S11_dB, S21_dB, S11_phase, S21_phase

    def printval(self):
        """
        Print the current values of the genetic algorithm parameters.
        """
        print(f"Population Size: {self.population_size}, Generations: {self.generations}, "
              f"Mutation Rate: {self.mutation_rate}, Max Mutation Attempts: {self.max_mutation_attempts}")




if __name__ == '__main__':
    rows, cols = 7, 11
    f_center = 0.8e12
    bandwidth = 0.4e12
    frequencies = np.arange(4e11, 1.2e12, 5e9)

    # Initialize Genetic Algorithm Optimizer
    ga_optimizer = GeneticAlgorithmOptimizer(rows, cols, f_center, bandwidth, frequencies)
    ga_optimizer.genetic_algorithm_search()
    ga_optimizer.visualize_best_grid()
    ga_optimizer.printval()
    