# Author: Luis Iracheta
# Artificial Intelligence Engineering
# Universidad Iberoamericana León

"""
Description:
This module implements the Differential Evolution (DE) algorithm, a
population-based evolutionary optimization method that operates on
real-valued vectors. It is designed for solving continuous and
nonlinear optimization problems without requiring gradient information.
"""


import numpy as np
import pandas as pd

class cl_alg_de():
    def __init__(self, function, population_size, num_variables, 
                 mutation_factor, crossover_rate, 
                 generations, limits, optimum):
        self.function = function
        self.population_size = population_size
        self.num_variables = num_variables
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.limits = limits
        self.optimum = optimum

    # The limits for the individuals are defined in especific ranges
    # Like: limits = [(min1, max1), (min2, max2), ...]
    def start_population(self, population_size, limits):
        population = []
        for i in range(population_size):
            individual = []
            for j in range(len(limits)):
                # Random value within the specified limits for each variable
                value = np.random.uniform(limits[j][0], limits[j][1])
                individual.append(value)
            population.append(individual)
        return population
    
    def evaluate_population(self, population): # Evaluate fitness of each individual
        fitness_values = []
        for individual in population:
            fitness = self.function(individual)
            fitness_values.append(fitness)
        return fitness_values
    
     # Differential Evolution mutation strategy
    def mutation_evol_diff(self, population, target_idx):
        indices = list(range(len(population)))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = []
        for i in range(len(population[0])):
            mutated_value = (population[a][i] + 
                             self.mutation_factor * 
                             (population[b][i] - population[c][i]))
            mutant_vector.append(mutated_value)
        return mutant_vector
    
    #  Crossover operation
    ''' In this method, we create a trial vector by combining the target vector
    and the mutant vector based on the crossover rate. For each dimension,'''
    def crossover(self, target_vector, mutant_vector):
        trial_vector = []
        for i in range(len(target_vector)):
            if np.random.rand() < self.crossover_rate:
                trial_vector.append(mutant_vector[i])
            else:
                trial_vector.append(target_vector[i])
        return trial_vector


    # Selection operation
    ''' This method selects between the target vector and the trial vector
    based on their fitness values. The one with the better fitness is chosen'''
    def selection(self, target_vector, trial_vector, optimum):   
        target_fitness = self.function(target_vector)
        trial_fitness = self.function(trial_vector)
        if optimum == 'min':
            if trial_fitness < target_fitness:
                return trial_vector
            else:
                return target_vector
        else:
            if trial_fitness > target_fitness:
                return trial_vector
            else:
                return target_vector
        
    def adjust_vector(self, vector):
        adjusted_vector = []
        for i in range(len(vector)):
            min_limit, max_limit = self.limits[i]
            adjusted_value = np.clip(vector[i], min_limit, max_limit)
            adjusted_vector.append(adjusted_value)
        return adjusted_vector
        
    def run(self):  # Main method to execute the DE algorithm
        population = self.start_population(self.population_size, self.limits)
        best_fitness_progress = []
        
        for gen in range(self.generations):
            new_population = []
            for i in range(self.population_size):
                target_vector = population[i]
                mutant_vector = self.mutation_evol_diff(population, i)
                trial_vector = self.crossover(target_vector, mutant_vector)
                trial_vector = self.adjust_vector(trial_vector)
                selected_vector = self.selection(target_vector, trial_vector, self.optimum)
                new_population.append(selected_vector)
            population = new_population
            
            fitness_values = self.evaluate_population(population)
            best_fitness = min(fitness_values)
            best_fitness_progress.append(best_fitness)
            #print(f"Generation {gen+1}: Best Fitness = {best_fitness}")
        
        best_index = np.argmin(self.evaluate_population(population))
        best_solution = population[best_index]
        #  Returning the best solution found
        return best_solution