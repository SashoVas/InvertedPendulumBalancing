

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
OUTPUT_COUNT = 1

config = NEATConfig(
    population_size=200,

    compatibility_threshold=3.0,

    survival_threshold=0.2,

    elitism_per_species=2,

    weight_mutation_rate=0.8,
    weight_perturb_rate=0.9,
    weight_perturb_scale=0.15,

    add_connection_rate=0.2,
    add_node_rate=0.1,

    toggle_connection_rate=0.05,
)


def evaluate_agent(genome):
    agent = NeatAgent(genome)
    return play_game(render=False, agent=agent)


def evaluate_population(population, pool, use_score=True):
    res = [(score, fitness)
           for score, fitness in pool.map(evaluate_agent, population.genomes)]
    best_score = max(score for score, _ in res)
    if use_score:
        return [score for score, _ in res], best_score
    else:
        return [fitness for _, fitness in res], best_score


def train_agents(config, use_score=True):
    population = NEATPopulation(
        input_count=INPUT_COUNT,
        output_count=OUTPUT_COUNT,
        config=config,
        seed=42
    )
    history = []

    with mp.Pool(processes=mp.cpu_count()-1) as pool:

        for generation in range(GENERATIONS):
            print(f"Starting generation {generation}:")
            generation_start = time.time()
            fitness, best_score = evaluate_population(
                population, pool, use_score=use_score)
            best = population.next_generation(fitness)
            history.append(best.fitness)
            print(
                f"Training finished in f{time.time() - generation_start: .2f}s")
            print("Best genome fitness:", best.fitness)
            print("Best genome score:", best_score)

    return best, history


if __name__ == "__main__":

    best_genome, history = train_agents(config, use_score=False)

    with open("agents/best_neat_genome_single_pendulum.pkl", "wb") as file:
        pickle.dump(best_genome, file)

    with open("agents/neat_history_single_pendulum.pkl", "wb") as file:
        pickle.dump(history, file)
