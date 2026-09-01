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

GENERATIONS = 250

EPISODES_PER_AGENT = 3

INPUT_COUNT = 8
OUTPUT_COUNT = 1

config = NEATConfig(
    population_size=500,
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


def evaluate_agent(args):
    """Average one genome over EPISODES_PER_AGENT rounds.

    The seeds depend only on the generation, so every genome in a generation
    faces the identical set of starts and their fitnesses are directly
    comparable. They change each generation, so nothing can overfit one
    particular starting nudge.
    """
    genome, generation, start_upright, episodes, mixed = args
    agent = NeatAgent(genome)

    results = [
        play_game(
            render=False,
            agent=agent,
            mode=1,
            start_upright=start_upright,
            seed=generation * episodes + episode,
        )
        for episode in range(episodes)
    ]
    if mixed:
        results += [
            play_game(render=False, agent=agent, mode=1, start_upright=False)
        ] * (round(episodes * 3))

    scores, fitnesses = zip(*results)
    return float(np.mean(scores)), float(np.mean(fitnesses))


def evaluate_population(
    population,
    pool,
    generation,
    use_score=True,
    start_upright=False,
    episodes=3,
    mixed=False,
):
    res = pool.map(
        evaluate_agent,
        [
            (genome, generation, start_upright, episodes, mixed)
            for genome in population.genomes
        ],
    )
    best_score = max(score for score, _ in res)
    if use_score:
        return [score for score, _ in res], best_score
    else:
        return [fitness for _, fitness in res], best_score


def train_agents(
    config,
    use_score=True,
    initial_population=None,
    start_upright=False,
    episodes=3,
    mixed=False,
):
    if initial_population is None:
        population = NEATPopulation(
            input_count=INPUT_COUNT, output_count=OUTPUT_COUNT, config=config, seed=42
        )
    else:
        population = initial_population
    history = []

    with mp.Pool(processes=mp.cpu_count() - 1) as pool:

        for generation in range(GENERATIONS):
            print(f"Starting generation {generation}:")
            generation_start = time.time()
            fitness, best_score = evaluate_population(
                population,
                pool,
                generation,
                use_score=use_score,
                start_upright=start_upright,
                episodes=episodes,
                mixed=mixed,
            )
            best = population.next_generation(fitness)
            history.append(best.fitness)
            print(
                f"Training finished in f{time.time() - generation_start: .2f}s")
            print("Spicies:", len(population.species))
            print("Best genome fitness:", best.fitness)
            print("Best genome score:", best_score)

    return best, history, population


if __name__ == "__main__":
    with open("agents/population_equal2.pkl", "rb") as f:
        population = pickle.load(f)
    GENERATIONS = 100

    best_genome, history, population = train_agents(
        config,
        use_score=False,
        initial_population=population,
        start_upright=True,
        episodes=4,
        mixed=True,
    )
    # with open("agents/upright.pkl", "wb") as file:
    #    pickle.dump(best_genome, file)
    # GENERATIONS = 200
    # best_genome, history, population = train_agents(
    #    config,
    #    use_score=False,
    #    initial_population=population,
    #    start_upright=False,
    #    episodes=1,
    # )
    with open("agents/best_neat_genome_double_pendulum_equal3.pkl", "wb") as file:
        pickle.dump(best_genome, file)

    with open("agents/neat_history_double_pendulum_equal3.pkl", "wb") as file:
        pickle.dump(history, file)

    with open("agents/population_equal3.pkl", "wb") as file:
        pickle.dump(population, file)
