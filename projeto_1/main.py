from src.algoritmo_genetico import *

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


def readBag(bagName):
    items = []
    
    print('Readed Items:')
    with open(f'./inputs/{bagName}', 'r') as file:
        while True:
            line = file.readline()
            
            if not line:
                break
            
            weight, value = line.strip().split(' ')

            items.append({
                'weight': int(weight),
                'value': int(value),
            })
            
            print(f'idx({len(items)}) ->\tweight: {weight};\tvalue: {value}')
    
    return items


def solve(bagData, num_individual, num_gens):
    ag = AG.create(num_individual = num_individual, num_chromosome = len(bagData['bag_items']), fitness = knapsack_problem)

    ag.generation_loop(num_gens = num_gens, metadata = bagData, log = False)
    
    print('\nExemplos de boas soluções:')
    for individual, fitness in ag.get_valid_chromosomes(bagData):
        selected_items = [ bagData['bag_items'][idx] for idx, value in enumerate(individual) if value == 1 ]
        
        total_weight = sum(item['weight'] for item in selected_items)
        total_value = sum(item['value'] for item in selected_items)
        
        print(individual, f"\tweight: {total_weight}, \tvalue: {total_value}, \tfitness: {fitness}")
    

def main():
    bagData = {
        'bag_items': readBag(input()),
        'max_weight': 100,
    }
    
    solve(bagData = bagData, num_individual = 10, num_gens = 10)
    

if __name__ == '__main__':
    main()