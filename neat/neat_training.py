

import pickle
import time

import numpy as np
import multiprocessing as mp

from game import play_game
from neat.neat import (
    NEATConfig,
    NEATPopulation,
    NeatAgent,
)

GENERATIONS = 100

INPUT_COUNT = 5
OUTPUT_COUNT = 3

config = NEATConfig(
    population_size=100,

    compatibility_threshold=3.0,

    survival_threshold=0.2,

    elitism_per_species=1,

    weight_mutation_rate=0.8,
    weight_perturb_rate=0.9,
    weight_perturb_scale=0.15,

    add_connection_rate=0.08,
    add_node_rate=0.03,

    toggle_connection_rate=0.01,
)


def evaluate_agent(agent):
    return play_game(render=False, agent=agent)


def evaluate_population(population, pool):
    agents = [NeatAgent(genome) for genome in population.genomes]
    return [score for score, _ in pool.map(evaluate_agent, agents)]


def train_agents(config):
    population = NEATPopulation(
        input_count=INPUT_COUNT,
        output_count=OUTPUT_COUNT,
        config=config,
        seed=42,
    )

    history = []

    with mp.Pool(processes=mp.cpu_count()-1) as pool:

        for generation in range(GENERATIONS):
            print(f"Starting generation {generation}:")
            generation_start = time.time()
            scores = evaluate_population(population, pool)
            best = population.next_generation(scores)
            history.append(best.fitness)
            print(
                f"Training finished in f{time.time() - generation_start: .2f}s")
            print("Best genome fitness:", best.fitness)

    return best, history


if __name__ == "__main__":

    best_genome, history = train_agents(config)

    with open("agents/best_neat_genome.pkl", "wb") as file:
        pickle.dump(best_genome, file)

    with open("agents/neat_history.pkl", "wb") as file:
        pickle.dump(history, file)
