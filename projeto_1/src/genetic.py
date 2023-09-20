from typing import Callable, Any, List, Union
from random import random, randint, choices
from time import time
from re import findall
import numpy as np


class AG:
    def __init__(self, metadata = {}, fitness: Callable[..., Any] = None):
        self.fitness = fitness
        self.fitness_historic = np.array([])
        self.population = np.array([])
        self.metadata = metadata

    @staticmethod
    def create(num_individual: int, num_chromosome: int, metadata = {}, fitness: Callable[..., Any] = None) -> None:
        this = AG(metadata, fitness)
        this.set_population(
            this.generate_random_population(num_individual, num_chromosome))
        
        return this

    @staticmethod
    def generate_random_chromosome(length: int, security_rate: int = 1) -> List[int]:
        # chromosome = np.array([])
        # for _ in range(length):
        #     random_number = randint(1, 100)
        #     chromosome = np.append(chromosome, random_number)
            
        # chromosome[chromosome <= security_rate] = 0
        # chromosome[chromosome > security_rate] = 1
        
        # print('chromosome', chromosome)
        
        # chromosome = np.random.randint(2, size=length) - np.random.randint(2, size=length)
        # chromosome[chromosome < 0] = 0

        # return chromosome
        chromosome = []
        for _ in range(length):
            random_number = randint(1,100)
            chromosome.append(1 if random_number <= security_rate else 0)
            
        return chromosome
            
        # return np.random.randint(2, size=length)

    def generate_random_population(self, num_individual: int, num_chromosome: int) -> List[List[int]]:
        self.num_individual = num_individual
        self.num_chromosome = num_chromosome
        
        return [ AG.generate_random_chromosome(num_chromosome) for _ in range(num_individual) ]

    def set_population(self, population: List[List[int]]) -> None:
        self.population = population

    def set_fitness(self, fitness: Callable[..., Any] = None) -> None:
        self.fitness = fitness

    def get_fitness(self, chromosome) -> Union[int, float]:
        return self.fitness(chromosome, self.metadata)

    def mutate(self, chromosome: List[int], mutate = .5) -> List[int]:
        if mutate <= random():
            return chromosome
        
        mutation_idx = randint(0, len(chromosome) - 1)
        chromosome[mutation_idx] = int(not chromosome[mutation_idx])

        return chromosome
    
    def mutate_population(self, mutate = .5) -> List[List[int]]:
        return self.mutate_individuals(self.population, mutate)
    
    def mutate_individuals(self, individuals: List[List[int]], mutate = .5) -> List[List[int]]:
        return [ self.mutate(individual, mutate) for individual in individuals ]
    
    def roulette(self, parents_with_fitness):
        def first(generic_list):
            return generic_list[0]
        
        def parent_drawer(parents, fitness_list):
            return first(choices(parents, weights = fitness_list, k = 1))

        if not bool(parents_with_fitness):
            return [], []
        
        parents, fitness_list = list(zip(*parents_with_fitness))
        
        father = parent_drawer(parents, fitness_list)
        mother = parent_drawer(parents, fitness_list)
        
        # print(father, mother)
        
        mother_idx = father_idx = None
        for idx, parent in enumerate(parents):
            if np.array_equal(parent, mother):
                mother_idx = idx
            if np.array_equal(parent, father):
                father_idx = idx
                
            if mother_idx is not None and father_idx is not None:
                break
        
        father_fitness = fitness_list[father_idx]
        mother_fitness = fitness_list[mother_idx]
        
        return [father, father_fitness], [mother, mother_fitness]
    
    def reproduce(self, parents, reproduce_meth = 'roulette'):
        childrens = []
        [father, father_fitness], [mother, mother_fitness] = getattr(self, reproduce_meth)(parents)
        
        section = randint(0, len(father) - 1)
        # section = len(father) // 2
        
        children = np.concatenate((father[:section], mother[section:]), axis=0)
        childrens.append(children)
        
        children = np.concatenate((mother[:section], father[section:]), axis=0)
        childrens.append(children)
            
        return np.array(childrens), [father, father_fitness], [mother, mother_fitness]
    
    
    def is_valid_chromosome(self, chromosome):
        return self.get_fitness(chromosome) >= 0
    
    def get_valid_chromosomes(self):
        individuals = []
        for chromosome in self.population:
            fitness = self.get_fitness(chromosome)
            
            if fitness >= .1:
                individuals.append([chromosome, fitness])
                
        return individuals
    
    def get_chromosomes_with_fitness(self):
        return map(lambda chromosome: [chromosome, self.get_fitness(chromosome)], self.population)


    def evaluate_parents_and_childrens(self, childrens, father_with_fitness, mother_with_fitness):
        population = self.population
        
        father, _ = father_with_fitness
        mother, _ = mother_with_fitness
        
        mother_idx = father_idx = None
        for idx, parent in enumerate(population):
            if np.array_equal(parent, mother):
                mother_idx = idx
            if np.array_equal(parent, father):
                father_idx = idx
                
            if mother_idx is not None and father_idx is not None:
                break
        
        candidates_with_fitness = list(map(lambda children: [children, self.get_fitness(children)], childrens))
        candidates_with_fitness.append(father_with_fitness)
        candidates_with_fitness.append(mother_with_fitness)
        candidates_with_fitness.sort(key=AG.valid_chromosome_compar, reverse=True)
        
        population[father_idx] = candidates_with_fitness[0][0]
        population[mother_idx] = candidates_with_fitness[1][0]
        
        return population

    
    @staticmethod
    def valid_chromosome_compar(individual):
        return individual[1]


    def evolve(self, reproduce_meth = 'roulette', mutate = .5):
        parents = sorted(self.get_chromosomes_with_fitness(), key=AG.valid_chromosome_compar, reverse=True)
        
        # while len(parents) < 2:
        #     print('Generating new population')
        #     self.set_population(
        #         self.mutate_individuals(self.population, 1))
        #     parents = sorted(self.get_valid_chromosomes(), key=AG.valid_chromosome_compar, reverse=True)
        
        childrens, father, mother = self.reproduce(parents, reproduce_meth)
        childrens = self.mutate_individuals(childrens, mutate)
        
        return self.evaluate_parents_and_childrens(childrens, father, mother)
        
        
    
    def average_chromosome_fitness(self):
        if not bool(self.population):
            return 0
        return sum(value for _, value in self.get_chromosomes_with_fitness()) / len(self.population)
    
    def generation_loop(self, num_gens, reproduce_meth = 'roulette', mutate = .5, log = False):
        average_fitness = self.average_chromosome_fitness()
        self.fitness_historic = [average_fitness]
        
        if log:
            print(f'Gen 0: {average_fitness}')
            # print(f'Population: {self.population}')
        
        for i in range(num_gens):
            self.set_population(self.evolve(reproduce_meth, mutate))
            average_fitness = self.average_chromosome_fitness()
            
            if log:
                print(f'Gen {i}: {average_fitness}')
                # print(f'Population: {population}')
                
            self.fitness_historic.append(average_fitness)
            
        return self.fitness_historic
    
    
    


def knapsack_problem(chromosome, metadata):
    total_weight = total_value = 0
    
    # print('knapsack_problem', metadata)
    
    max_weight = metadata['max_weight']
    bag_items = metadata['bag_items']
    
    for idx, value in enumerate(chromosome):
        total_weight += (value * bag_items[idx]['weight'])    
        total_value += (value * bag_items[idx]['value'])  
        
    if max_weight < total_weight:
        return 1e-10
        
    return total_value    


def read_bag(bag_filename):
    items = []
    
    print('Readed Items:')
    with open(bag_filename, 'r') as file:
        lines = file.readlines()
        
    num_elements = int(lines[0].strip())
    capacity = int(lines[-1].strip())
    
    print(f'Capacity {capacity}')
    print(f'Number of elements {num_elements}')
    for line in lines[1:-1]:
        numbers = findall(r"[0-9]+", line)
        
        weight, value = int(numbers[1]), int(numbers[2])
        
        items.append({
            'weight': int(weight),
            'value': int(value),
        })
        
        print(f'idx({int(numbers[0])}) ->\tweight: {weight};\tvalue: {value}')
    
    return {
        'bag_items': items,
        'max_weight': capacity,
        'num_elements': num_elements,
    }


def solve_knapsack_problem_ag(bag_problem_filename):
    num_individual, num_gens = 1000, 100
    
    bag_data = read_bag(bag_problem_filename)
    ag = AG.create(num_individual = num_individual, num_chromosome = bag_data['num_elements'], metadata = bag_data, fitness = knapsack_problem)

    ag.generation_loop(num_gens = num_gens, log = True)
    
    print('\nExemplos de boas soluções:')
    
    max_value = max_weight = 0
    for individual, fitness in ag.get_valid_chromosomes():
        selected_items = [ bag_data['bag_items'][idx] for idx, value in enumerate(individual) if value == 1 ]
        
        total_weight = sum(item['weight'] for item in selected_items)
        total_value = sum(item['value'] for item in selected_items)
        
        if max_value < total_value:
            max_value = total_value
            
        if max_weight < total_weight:
            max_weight = total_weight
        
        print(individual, f"\tweight: {total_weight}, \tvalue: {total_value}, \tfitness: {fitness}")
        
    
    output_line = f"Instancia {bag_problem_filename} : {max_value}, {max_weight}\n"
    
    with open("outputs/genetic.out", "a+") as output_file:
        output_file.write(output_line)
        
    return max_value
    

def init_genetic_algorithm(bag_problem_filename):
    return solve_knapsack_problem_ag(bag_problem_filename)           
            
def rate_genetic_algorithm(input_file_path):
    start_time = time()
    print(f"Genetic Algorithm start")
    result = init_genetic_algorithm(input_file_path)
    execution_time = time() - start_time
    print(f"Execution time: {execution_time} seconds")
    
    return result