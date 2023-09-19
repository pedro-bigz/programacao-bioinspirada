from src.genetic import *
from src.dynamic import *
import matplotlib.pyplot as plt
import re

def knapsack_problem(chromosome, metadata):
    total_weight = total_value = 0
    
    # print('knapsack_problem', metadata)
    
    max_weight = metadata['max_weight']
    bag_items = metadata['bag_items']
    
    for idx, value in enumerate(chromosome):
        total_weight += (value * bag_items[idx]['weight'])    
        total_value += (value * bag_items[idx]['value'])  
        
    if max_weight < total_weight:
        return -1
        
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
        numbers = re.findall(r"[0-9]+", line)
        
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
    num_individual, num_gens = 100, 10000
    
    bag_data = read_bag(bag_problem_filename)
    ag = AG.create(num_individual = num_individual, num_chromosome = bag_data['num_elements'], metadata = bag_data, fitness = knapsack_problem)

    ag.generation_loop(num_gens = num_gens, log = True)
    
    print('\nExemplos de boas soluções:')
    
    max_value = 0
    for individual, fitness in ag.get_valid_chromosomes():
        selected_items = [ bag_data['bag_items'][idx] for idx, value in enumerate(individual) if value == 1 ]
        
        total_weight = sum(item['weight'] for item in selected_items)
        total_value = sum(item['value'] for item in selected_items)
        
        if max_value < total_value:
            max_value = total_value
        
        print(individual, f"\tweight: {total_weight}, \tvalue: {total_value}, \tfitness: {fitness}")
        
    
    output_line = f"Instancia {bag_problem_filename} : {max_value}\n"
    
    with open("outputs/genetic.out", "a+") as output_file:
        output_file.write(output_line)
        
    return max_value
    

def init_genetic_algorithm(bag_problem_filename):
    return solve_knapsack_problem_ag(bag_problem_filename)           
            
def rate_genetic_algorithm(input_file_path):
    start_time = time.time()
    print(f"Genetic Algorithm start")
    result = init_genetic_algorithm(input_file_path)
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time} seconds")
    
    return result


def plot_results(bag_problem_filename, result_ag, result_dy):
    labels = np.array(["Genetic Algorithm", "Dynamic Programming"])
    values = np.array([result_ag, result_dy])
    
    plt.bar(labels, values)
    plt.show()
    # plt.savefig(bag_problem_filename)
    

def main(bag_problem_filename):
    result_ag = rate_genetic_algorithm(bag_problem_filename)    
    result_dy = rate_dynamic_programming(bag_problem_filename)
    
    plot_results(bag_problem_filename, result_ag, result_dy)

if __name__ == '__main__':
    main(f'./inputs/input{input()}.in')