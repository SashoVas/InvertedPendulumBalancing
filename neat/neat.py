from __future__ import annotations
import pygame
from dataclasses import dataclass, field
from enum import Enum
import copy
import heapq
import pickle
from collections import deque

import numpy as np


@dataclass
class NEATConfig:
    population_size: int = 150

    # Speciation
    compatibility_threshold: float = 3.0
    compatibility_c1: float = 1.0
    compatibility_c2: float = 1.0
    compatibility_c3: float = 0.4

    # Reproduction
    survival_threshold: float = 0.2
    elitism_per_species: int = 1

    # Weight mutation
    weight_mutation_rate: float = 0.8
    weight_perturb_rate: float = 0.9
    weight_perturb_scale: float = 0.15
    weight_reset_scale: float = 1.0

    # Structural mutation
    add_connection_rate: float = 0.08
    add_node_rate: float = 0.03
    toggle_connection_rate: float = 0.01

    # Network
    activation_weight_limit: float = 5.0


class NodeType(Enum):
    INPUT = 0
    BIAS = 1
    HIDDEN = 2
    OUTPUT = 3


@dataclass
class NodeGene:
    id: int
    type: NodeType


@dataclass
class ConnectionGene:
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True


class InnovationTracker:

    def __init__(self, first_dynamic_node_id: int):
        self.next_innovation = 0
        self.next_node_id = first_dynamic_node_id

        self.connection_innovations = {}
        self.split_nodes = {}

    def get_connection_innovation(self, in_node: int, out_node: int):
        key = (in_node, out_node)

        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_innovation
            self.next_innovation += 1

        return self.connection_innovations[key]

    def get_connection_innovation_if_exists(self, in_node: int, out_node: int):
        key = (in_node, out_node)
        return self.connection_innovations.get(key, None)

    def get_split_node(self, connection_innovation: int):
        if connection_innovation not in self.split_nodes:
            node_id = self.next_node_id
            self.next_node_id += 1

            self.split_nodes[connection_innovation] = node_id

        return self.split_nodes[connection_innovation]


class Genome:

    def __init__(self):
        self.nodes: dict[int, NodeGene] = {}
        self.connections: dict[int, ConnectionGene] = {}
        self.fitness = -np.inf

    @classmethod
    def initial(
        cls,
        input_count: int,
        output_count: int,
        tracker: InnovationTracker,
        rng: np.random.Generator,
    ):
        genome = cls()

        input_ids = list(range(input_count))
        for node_id in input_ids:
            genome.nodes[node_id] = NodeGene(node_id, NodeType.INPUT)

        bias_id = input_count
        genome.nodes[bias_id] = NodeGene(bias_id, NodeType.BIAS)

        output_start = input_count + 1
        output_ids = list(range(output_start, output_start + output_count))
        for node_id in output_ids:
            genome.nodes[node_id] = NodeGene(node_id, NodeType.OUTPUT)

        sources = input_ids + [bias_id]
        for source in sources:
            for target in output_ids:
                innovation = tracker.get_connection_innovation(source, target)
                genome.connections[innovation] = ConnectionGene(
                    innovation=innovation,
                    in_node=source,
                    out_node=target,
                    weight=float(rng.normal(0.0, 0.5)),
                    enabled=True,
                )

        return genome

    def copy(self):
        return copy.deepcopy(self)

    def distance(self, other: "Genome", config: NEATConfig):

        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        if not innovations1 and not innovations2:
            return 0.0

        max1 = max(innovations1, default=-1)
        max2 = max(innovations2, default=-1)
        max_common_range = min(max1, max2)
        matching = innovations1 & innovations2

        disjoint = 0
        excess = 0

        for innovation in innovations1 ^ innovations2:
            if innovation > max_common_range:
                excess += 1
            else:
                disjoint += 1

        weight_difference = 0.0

        if matching:
            weight_difference = np.mean(
                [
                    abs(self.connections[i].weight - other.connections[i].weight)
                    for i in matching
                ]
            )

        n = max(len(self.connections), len(other.connections))
        n = max(n, 1)

        return (
            config.compatibility_c1 * excess / n
            + config.compatibility_c2 * disjoint / n
            + config.compatibility_c3 * weight_difference
        )

    def creates_cycle(self, source: int, target: int):
        adjacency = {}

        for connection in self.connections.values():
            if not connection.enabled:
                continue

            adjacency.setdefault(connection.in_node, []).append(connection.out_node)

        stack = [target]
        visited = set()

        while stack:
            current = stack.pop()
            if current == source:
                return True
            if current in visited:
                continue
            visited.add(current)
            for next_node in adjacency.get(current, []):
                stack.append(next_node)

        return False

    def mutate_weights(self, rng: np.random.Generator, config: NEATConfig):
        for connection in self.connections.values():
            if rng.random() > config.weight_mutation_rate:
                continue
            if rng.random() < config.weight_perturb_rate:
                connection.weight += rng.normal(0.0, config.weight_perturb_scale)
            else:
                connection.weight = rng.normal(0.0, config.weight_reset_scale)

            connection.weight = float(
                np.clip(
                    connection.weight,
                    -config.activation_weight_limit,
                    config.activation_weight_limit,
                )
            )

    def mutate_add_connection(
        self, tracker: InnovationTracker, rng: np.random.Generator
    ):

        sources = [
            node.id for node in self.nodes.values() if node.type != NodeType.OUTPUT
        ]

        targets = [
            node.id
            for node in self.nodes.values()
            if node.type in (NodeType.HIDDEN, NodeType.OUTPUT)
        ]

        if not sources or not targets:
            return

        for _ in range(40):

            source = int(rng.choice(sources))
            target = int(rng.choice(targets))

            if source == target:
                continue

            already_exists = tracker.get_connection_innovation_if_exists(source, target)
            if already_exists is not None and already_exists in self.connections:
                continue

            if self.creates_cycle(source, target):
                continue

            innovation = tracker.get_connection_innovation(source, target)

            self.connections[innovation] = ConnectionGene(
                innovation=innovation,
                in_node=source,
                out_node=target,
                weight=float(rng.normal(0.0, 0.5)),
                enabled=True,
            )

            return

    def mutate_add_node(self, tracker: InnovationTracker, rng: np.random.Generator):

        candidates = [
            connection
            for connection in self.connections.values()
            if connection.enabled
            and tracker.get_split_node(connection.innovation) not in self.nodes
        ]

        if not candidates:
            return

        connection = candidates[int(rng.integers(len(candidates)))]
        connection.enabled = False

        new_node_id = tracker.get_split_node(connection.innovation)
        self.nodes[new_node_id] = NodeGene(new_node_id, NodeType.HIDDEN)

        innovation_1 = tracker.get_connection_innovation(
            connection.in_node, new_node_id
        )
        self.connections[innovation_1] = ConnectionGene(
            innovation=innovation_1,
            in_node=connection.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True,
        )

        innovation_2 = tracker.get_connection_innovation(
            new_node_id, connection.out_node
        )
        self.connections[innovation_2] = ConnectionGene(
            innovation=innovation_2,
            in_node=new_node_id,
            out_node=connection.out_node,
            weight=connection.weight,
            enabled=True,
        )

    def mutate_toggle_connection(
        self,
        rng: np.random.Generator,
    ):
        if not self.connections:
            return

        connection = self.connections[int(rng.choice(list(self.connections.keys())))]
        connection.enabled = not connection.enabled

    def mutate(
        self,
        tracker: InnovationTracker,
        rng: np.random.Generator,
        config: NEATConfig,
    ):

        self.mutate_weights(rng, config)

        if rng.random() < config.add_connection_rate:
            self.mutate_add_connection(tracker, rng)

        if rng.random() < config.add_node_rate:
            self.mutate_add_node(tracker, rng)

        if rng.random() < config.toggle_connection_rate:
            self.mutate_toggle_connection(rng)

    @staticmethod
    def crossover(parent1: Genome, parent2: Genome, rng: np.random.Generator):

        # Parent1 must be the fitter parent.
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1
        elif parent1.fitness == parent2.fitness:
            if rng.random() < 0.5:
                parent1, parent2 = parent2, parent1

        child = Genome()

        for node_id, node in parent1.nodes.items():
            child.nodes[node_id] = copy.deepcopy(node)

        all_innovations = sorted(set(parent1.connections) | set(parent2.connections))

        for innovation in all_innovations:

            gene1 = parent1.connections.get(innovation)
            gene2 = parent2.connections.get(innovation)

            chosen = None

            # Matching gene
            if gene1 is not None and gene2 is not None:

                chosen = copy.deepcopy(gene1 if rng.random() < 0.5 else gene2)

                # NEAT convention:
                # if either parent disabled it,
                # child has a high probability of disabled.
                if not gene1.enabled or not gene2.enabled:
                    chosen.enabled = rng.random() >= 0.75

            # Excess/disjoint genes only come from fitter parent.
            elif gene1 is not None:
                chosen = copy.deepcopy(gene1)

            if chosen is not None:
                child.connections[innovation] = chosen

        # Add any required nodes for inherited connections.
        for connection in child.connections.values():

            if connection.in_node not in child.nodes:
                child.nodes[connection.in_node] = copy.deepcopy(
                    parent1.nodes.get(connection.in_node)
                    or parent2.nodes[connection.in_node]
                )

            if connection.out_node not in child.nodes:
                child.nodes[connection.out_node] = copy.deepcopy(
                    parent1.nodes.get(connection.out_node)
                    or parent2.nodes[connection.out_node]
                )

        return child


class NeatNetwork:

    def __init__(self, genome: Genome):

        self.genome = genome

        self.input_nodes = sorted(
            node.id for node in genome.nodes.values() if node.type == NodeType.INPUT
        )

        self.bias_nodes = [
            node.id for node in genome.nodes.values() if node.type == NodeType.BIAS
        ]

        self.output_nodes = sorted(
            node.id for node in genome.nodes.values() if node.type == NodeType.OUTPUT
        )

        self.incoming = {}

        for connection in genome.connections.values():

            if not connection.enabled:
                continue

            self.incoming.setdefault(connection.out_node, []).append(connection)

        self.order = self._topological_sort()

    def _topological_sort(self):
        indegree = {node_id: 0 for node_id in self.genome.nodes}

        adjacency = {}

        for connection in self.genome.connections.values():
            if not connection.enabled:
                continue
            adjacency.setdefault(connection.in_node, []).append(connection.out_node)
            indegree[connection.out_node] += 1

        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)

        order = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)

            for next_node in adjacency.get(node_id, []):
                indegree[next_node] -= 1
                if indegree[next_node] == 0:
                    queue.append(next_node)

        return order

    def activate(self, inputs):
        values = {}

        for node_id, value in zip(self.input_nodes, inputs):
            values[node_id] = float(value)

        # Bias
        for node_id in self.bias_nodes:
            values[node_id] = 1.0

        # Hidden + output nodes
        for node_id in self.order:
            node = self.genome.nodes[node_id]

            if node.type in (NodeType.INPUT, NodeType.BIAS):
                continue

            total = 0.0

            for connection in self.incoming.get(node_id, []):
                total += values[connection.in_node] * connection.weight

            values[node_id] = np.tanh(total)

        return np.array(
            [values.get(node_id, 0) for node_id in self.output_nodes], dtype=np.float32
        )


@dataclass
class Species:
    id: int
    representative: Genome
    members: list[Genome] = field(default_factory=list)


class NEATPopulation:

    def __init__(
        self,
        input_count: int,
        output_count: int,
        config: NEATConfig | None = None,
        seed: int | None = None,
    ):

        self.config = config or NEATConfig()
        self.rng = np.random.default_rng(seed)
        first_hidden_id = input_count + 1 + output_count
        self.input_count = input_count
        self.output_count = output_count
        self.tracker = InnovationTracker(first_dynamic_node_id=first_hidden_id)

        self.genomes = [
            Genome.initial(self.input_count, self.output_count, self.tracker, self.rng)
            for _ in range(self.config.population_size)
        ]

        self.species = []
        self.next_species_id = 0

    def speciate(self):

        for species in self.species:
            species.members.clear()

        for genome in self.genomes:
            found_species = False
            for species in self.species:
                distance = genome.distance(species.representative, self.config)
                if distance < self.config.compatibility_threshold:
                    species.members.append(genome)
                    found_species = True
                    break

            if not found_species:
                species = Species(
                    id=self.next_species_id,
                    representative=genome.copy(),
                    members=[genome],
                )
                self.next_species_id += 1
                self.species.append(species)

        # Remove empty species
        self.species = [species for species in self.species if species.members]

        # Pick a new representative.
        for species in self.species:
            index = int(self.rng.integers(len(species.members)))
            species.representative = species.members[index].copy()

    def assign_fitness(self, fitnesses):
        for genome, fitness in zip(self.genomes, fitnesses):
            genome.fitness = float(fitness)

    def best_genome(self):
        return max(self.genomes, key=lambda genome: genome.fitness)

    def _allocate_offspring(self, remaining):

        if remaining <= 0:
            return [0] * len(self.species)

        scores = []

        for species in self.species:
            adjusted = np.mean([genome.fitness for genome in species.members])
            scores.append(max(adjusted, 0.0))

        scores = np.asarray(scores, dtype=float)

        if scores.sum() <= 0:
            scores[:] = 1.0

        raw = scores / scores.sum() * remaining
        allocation = np.floor(raw).astype(int)
        leftover = remaining - int(allocation.sum())

        if leftover > 0:
            fractions = raw - allocation
            order = np.argsort(fractions)[::-1]
            for i in order[:leftover]:
                allocation[i] += 1

        return allocation.tolist()

    def reproduce(self):
        new_genomes = []
        elites_per_species = []

        for species in self.species:
            ranked = sorted(
                species.members, key=lambda genome: genome.fitness, reverse=True
            )
            elite_count = min(self.config.elitism_per_species, len(ranked))
            elites_per_species.append(ranked[:elite_count])

        total_elites = sum(len(elites) for elites in elites_per_species)

        remaining = self.config.population_size - total_elites

        allocation = self._allocate_offspring(remaining)

        # Elites
        for elites in elites_per_species:
            for genome in elites:
                new_genomes.append(genome.copy())

        # Children
        for species_index, species in enumerate(self.species):
            ranked = sorted(
                species.members, key=lambda genome: genome.fitness, reverse=True
            )

            parent_count = max(
                1, int(np.ceil(len(ranked) * self.config.survival_threshold))
            )

            parents = ranked[:parent_count]
            child_count = allocation[species_index]

            for _ in range(child_count):
                p1 = parents[int(self.rng.integers(len(parents)))]
                p2 = parents[int(self.rng.integers(len(parents)))]
                child = Genome.crossover(p1, p2, self.rng)
                child.mutate(self.tracker, self.rng, self.config)
                new_genomes.append(child)

        if len(new_genomes) < self.config.population_size:
            new_genomes = new_genomes + [
                Genome.initial(
                    self.input_count, self.output_count, self.tracker, self.rng
                )
                for _ in range(self.config.population_size - len(new_genomes))
            ]

        self.genomes = new_genomes[: self.config.population_size]

    def next_generation(self, fitnesses):

        self.assign_fitness(fitnesses)

        self.speciate()

        best = self.best_genome()

        self.reproduce()

        return best


class NeatAgent:

    def __init__(self, genome: Genome):
        self.genome = genome
        self.network = NeatNetwork(genome)

    @staticmethod
    def load_from_file(file_name):
        with open(file_name, "rb") as f:
            obj = pickle.load(f)

        return NeatAgent(obj)

    def get_action(self, state, use_force=False):

        output = self.network.activate(state)
        if use_force:
            return float(np.clip(output[0], -2.0, 2.0))
        action = np.argmax(output).item()

        if action == 0:
            action = {pygame.K_RIGHT: False, pygame.K_LEFT: False}
        elif action == 1:
            action = {pygame.K_RIGHT: True, pygame.K_LEFT: False}
        else:
            action = {pygame.K_RIGHT: False, pygame.K_LEFT: True}

        return action
