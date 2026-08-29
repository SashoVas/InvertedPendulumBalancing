import torch
from collections import OrderedDict
from neuroevolution.agent import Agent, AgentNeuralNetwork, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE
import multiprocessing as mp
import time
import numpy as np
from game import play_game

GENERATIONS = 50
POPULATION_SIZE = 100
ELITISM_COUNT = 5
NEW_SAMPLE_COUNT = 20
MUTATION_RATE = 0.03
MUTATION_STRENGTH = 0.1


def generate_state_dicts(count):
    results = []
    for _ in range(count):
        state_dict = {}

        random_weights_input = torch.randn(
            (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE))
        random_weights_hidden = torch.randn(
            (HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))
        random_weights_output = torch.randn(
            (OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE))
        state_dict = OrderedDict([('neural_network.0.weight', random_weights_input),
                                  ('neural_network.2.weight',
                                   random_weights_hidden),
                                  ('neural_network.4.weight',
                                   random_weights_output),
                                  ('neural_network.0.bias',
                                   torch.randn(HIDDEN_LAYER_SIZE)),
                                  ('neural_network.2.bias',
                                   torch.randn(HIDDEN_LAYER_SIZE)),
                                  ('neural_network.4.bias', torch.randn(OUTPUT_LAYER_SIZE))])

        results.append(state_dict)
    return results


def evaluate_agent(agent):
    return play_game(render=False, agent=agent)


def rank_based_selection(population, fitness_scores, num_selections):
    sorted_indices = np.argsort(fitness_scores)

    ranks = np.arange(1, len(fitness_scores)+1)

    probs = ranks / np.sum(ranks)

    chosen = np.random.choice(
        np.array(population)[sorted_indices], size=num_selections, p=probs)
    return chosen.tolist()


def selection(agents, scores):
    agents = [
        agent
        for _, agent in sorted(
            zip(scores, agents),
            key=lambda x: x[0],
            reverse=True
        )
    ]
    elitism = agents[:ELITISM_COUNT]
    state_dicts = generate_state_dicts(NEW_SAMPLE_COUNT)
    new_samples = [Agent(state_dict=state_dicts[i])
                   for i in range(NEW_SAMPLE_COUNT)]
    return elitism, new_samples + rank_based_selection(agents, scores, POPULATION_SIZE - NEW_SAMPLE_COUNT)


def blend_crossover(parent1_state_dict, parent2_state_dict):
    child = OrderedDict()

    for key in parent1_state_dict.keys():
        p1 = parent1_state_dict[key]
        p2 = parent2_state_dict[key]

        alpha = torch.rand_like(p1)

        child[key] = alpha * p1 + (1 - alpha) * p2

    return child


def crossover(agents, elitism):
    parents = agents + elitism

    children = []

    while len(children) < POPULATION_SIZE - ELITISM_COUNT:
        parent1, parent2 = np.random.choice(parents, 2, replace=False)
        child_state_dict = blend_crossover(
            parent1.state_dict(), parent2.state_dict())
        children.append(Agent(state_dict=child_state_dict))

    return children


def mutate(agent, mutation_rate=0.01, mutation_strength=0.1):
    with torch.no_grad():
        for param in agent.model.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_strength

            param.add_(mask * noise)


def mutate_agents(agents, mutation_rate=0.01, mutation_strength=0.1):
    for agent in agents:
        mutate(agent, mutation_rate, mutation_strength)
    return agents


def train_agents(epochs, count):
    state_dicts = generate_state_dicts(count)
    agents = [Agent(state_dict=state_dicts[i]) for i in range(count)]

    start = time.time()
    with mp.Pool(processes=mp.cpu_count()-1) as pool:
        for i in range(epochs):
            print(f"Evaluating agents for epoch {i+1}/{epochs}...")
            scores = [score for score, _ in pool.map(evaluate_agent, agents)]
            fitness = [fitness for _, fitness in pool.map(
                evaluate_agent, agents)]

            elitism, agents = selection(agents, fitness)

            agents = crossover(agents, elitism)
            agents = mutate_agents(
                agents, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH)
            agents = elitism + agents
            print(f"Epoch {i+1}/{epochs} completed. Best score: {max(scores)}")

    end = time.time()
    print(f"Training completed in {end - start:.2f} seconds.")

    elitism[0].save("agents/neuroevolution.pth")
    return agents


if __name__ == "__main__":
    mp.freeze_support()

    agents = train_agents(GENERATIONS, POPULATION_SIZE)
