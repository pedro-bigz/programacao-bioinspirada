from src.genetic import *
from src.dynamic import *
import matplotlib.pyplot as plt


def plot_results(result_ag, result_dy):
    labels = np.array(["Genetic Algorithm", "Dynamic Programming"])
    values = np.array([result_ag, result_dy])
    
    plt.bar(labels, values)
    plt.show()
    # plt.savefig(bag_problem_filename)
    

def main(bag_problem_filename):
    result_ag = rate_genetic_algorithm(bag_problem_filename)    
    result_dy = rate_dynamic_programming(bag_problem_filename)
    
    plot_results(result_ag, result_dy)

if __name__ == '__main__':
    for i in input().split(' '):
        main(f'./inputs/input{i}.in')