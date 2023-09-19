from typing import Callable, Any, List, Union
from random import random, randint
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
    def generate_random_chromosome(length: int) -> List[int]:
        return np.random.randint(2, size=length)

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
    
    def roulette(self, parents):
        def parentDrawer(fitness_list):
            roulette, accum, sorted_value = ([], 0, random())
            fitness_sum = sum(value for value in fitness_list if value is not None)
            
            for idx, value in enumerate(fitness_list):
                if value is not None:
                    accum += value
                    roulette.append(accum / fitness_sum)
                    
                    if roulette[-1] >= sorted_value:
                        return idx

        if not bool(parents):
            return [], []
        
        individuals, fitness_list = list(zip(*parents))
        
        idx_father = parentDrawer(fitness_list)
        idx_mother = parentDrawer([ value if key != idx_father else None for key, value in enumerate(fitness_list) ])
        
        return individuals[idx_father], individuals[idx_mother]
    
    def reproduce(self, parents, reproduce_meth = 'roulette'):
        childrens = []
        while len(childrens) < len(self.population):
            father, mother = getattr(self, reproduce_meth)(parents)
            
            # section = randint(0, len(father) - 1)
            section = len(father) // 2
            
            children = np.concatenate((father[:section], mother[section:]), axis=0)
            childrens.append(children)
            
        return np.array(childrens)
    
    
    def is_valid_chromosome(self, chromosome):
        return self.get_fitness(chromosome) >= 0
    
    def get_valid_chromosomes(self):
        individuals = []
        for chromosome in self.population:
            fitness = self.get_fitness(chromosome)
            
            if fitness >= 0:
                individuals.append([chromosome, fitness])
                
        return individuals
    
    @staticmethod
    def valid_chromosome_compar(individual):
        return individual[1]

    def evolve(self, reproduce_meth = 'roulette', mutate = .5):
        parents = sorted(self.get_valid_chromosomes(), key=AG.valid_chromosome_compar, reverse=True)
        
        while len(parents) < 2:
            print('Generating new population')
            self.set_population(
                self.generate_random_population(self.num_individual, self.num_chromosome))
            parents = sorted(self.get_valid_chromosomes(), key=AG.valid_chromosome_compar, reverse=True)
        
        childrens = self.reproduce(parents, reproduce_meth)
        
        return self.mutate_individuals(childrens, mutate)
    
    def average_chromosome_fitness(self):
        if not bool(self.population):
            return 0
        return sum(value for _, value in self.get_valid_chromosomes()) / len(self.population)
    
    def generation_loop(self, num_gens, reproduce_meth = 'roulette', mutate = .5, log = False):
        average_fitness = self.average_chromosome_fitness()
        self.fitness_historic = [average_fitness]
        
        if log:
            print(f'Gen 0: {average_fitness}')
            # print(f'Population: {self.population}')
        
        for i in range(num_gens):
            population = self.evolve(reproduce_meth, mutate)
            
            if log:
                print(f'Gen {i}: {average_fitness}')
                # print(f'Population: {population}')
                
            self.set_population(population)
            self.fitness_historic.append(self.average_chromosome_fitness())
            
        return self.fitness_historic