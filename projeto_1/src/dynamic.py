import numpy as np
import re
import time

def knapsack(size, value, weight, capacity, dp):
    if size == 0 or capacity == 0:
        return 0
    if dp[size - 1][capacity] != -1:
        return dp[size - 1][capacity]
    if weight[size - 1] > capacity:
        dp[size - 1][capacity] = knapsack(size - 1, value, weight, capacity, dp)
        return dp[size - 1][capacity]
    
    a = value[size - 1] + knapsack(size - 1, value, weight, capacity - weight[size - 1], dp)
    b = knapsack(size - 1, value, weight, capacity, dp)
    dp[size - 1][capacity] = max(a, b)
    
    return dp[size - 1][capacity]

def solve_knapsack_problem(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    n = int(lines[0].strip())
    capacity = int(lines[-1].strip())
    
    id, value, weight = [], [], []
    for line in lines[1:-1]:
        numbers = re.findall(r"[0-9]+", line)
        id.append(int(numbers[0]) - 1)
        value.append(int(numbers[1]))
        weight.append(int(numbers[2]))
    
    dp = np.full((n, capacity + 1), -1, dtype=int)
    max_value = knapsack(n, value, weight, capacity, dp)
    return max_value

def init_dynamic_programming(input_file_path):
    max_value = solve_knapsack_problem(input_file_path)
    output_line = f"Instancia {input_file_path} : {max_value}\n"
    
    with open("outputs/dynamic.out", "a+") as output_file:
        output_file.write(output_line)
        
    return max_value
            
            
def rate_dynamic_programming(input_file_path):
    start_time = time.time()
    print(f"Dynamic Programming start")
    result = init_dynamic_programming(input_file_path)
    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time} seconds")
    
    return result
