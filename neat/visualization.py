import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_genome(
    genome,
    *,
    title="Genome",
    figsize=(14, 8),
    show=True,
    save_path=None,
):
    enabled_connections = [
        connection
        for connection in genome.connections.values()
        if connection.enabled
    ]

    outgoing = {}
    incoming = {}

    for connection in enabled_connections:
        outgoing.setdefault(
            connection.in_node, []
        ).append(connection.out_node)

        incoming.setdefault(
            connection.out_node, []
        ).append(connection.in_node)

    input_nodes = sorted(
        node.id
        for node in genome.nodes.values()
        if node.type.name == "INPUT"
    )

    bias_nodes = sorted(
        node.id
        for node in genome.nodes.values()
        if node.type.name == "BIAS"
    )

    hidden_nodes = sorted(
        node.id
        for node in genome.nodes.values()
        if node.type.name == "HIDDEN"
    )

    output_nodes = sorted(
        node.id
        for node in genome.nodes.values()
        if node.type.name == "OUTPUT"
    )

    layers = {
        node_id: 0
        for node_id in input_nodes + bias_nodes
    }

    remaining = set(hidden_nodes + output_nodes)

    while remaining:

        progress = False

        for node_id in list(remaining):

            predecessors = incoming.get(
                node_id,
                []
            )

            if not all(
                predecessor in layers
                for predecessor in predecessors
            ):
                continue

            if predecessors:
                layers[node_id] = (
                    max(
                        layers[p]
                        for p in predecessors
                    ) + 1
                )
            else:
                layers[node_id] = 1

            remaining.remove(node_id)
            progress = True

        if not progress:
            raise RuntimeError(
                "Cannot determine network layers. "
                "The genome may contain a cycle."
            )

    max_hidden_layer = max(
        (
            layers[node_id]
            for node_id in hidden_nodes
        ),
        default=0,
    )

    output_layer = max_hidden_layer + 1

    for node_id in output_nodes:
        layers[node_id] = output_layer

    nodes_by_layer = {}

    for node_id, layer in layers.items():
        nodes_by_layer.setdefault(
            layer, []
        ).append(node_id)

    for node_ids in nodes_by_layer.values():
        node_ids.sort()

    positions = {}

    x_spacing = 3.5

    for layer, node_ids in nodes_by_layer.items():

        count = len(node_ids)

        if count == 1:
            y_positions = [0.0]
        else:
            total_height = max(
                3.0,
                (count - 1) * 1.5
            )

            y_positions = [
                total_height / 2
                - i * (
                    total_height / (count - 1)
                )
                for i in range(count)
            ]

        x = layer * x_spacing

        for node_id, y in zip(
            node_ids,
            y_positions
        ):
            positions[node_id] = (
                x,
                y
            )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    fig.patch.set_facecolor("#f7f8fc")
    ax.set_facecolor("#f7f8fc")

    if enabled_connections:
        max_weight = max(
            abs(connection.weight)
            for connection in enabled_connections
        )
    else:
        max_weight = 1.0

    # Prevent division by zero.
    max_weight = max(
        max_weight,
        1e-8
    )

    norm = Normalize(
        vmin=-max_weight,
        vmax=max_weight
    )

    cmap = plt.get_cmap("RdBu")

    for connection in enabled_connections:

        if (
            connection.in_node not in positions
            or connection.out_node not in positions
        ):
            continue

        x1, y1 = positions[
            connection.in_node
        ]

        x2, y2 = positions[
            connection.out_node
        ]

        weight = connection.weight
        connection_color = cmap(
            norm(weight)
        )

        magnitude = abs(weight)

        linewidth = (
            1.0
            + 5.0 * magnitude / max_weight
        )

        rad = 0.0

        if (
            connection.in_node
            in incoming.get(
                connection.out_node,
                []
            )
        ):
            rad = (
                0.08
                if (
                    connection.in_node
                    + connection.out_node
                ) % 2 == 0
                else -0.08
            )

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=linewidth,
            color=connection_color,
            alpha=0.85,
            connectionstyle=(
                f"arc3,rad={rad}"
            ),
            shrinkA=22,
            shrinkB=22,
            zorder=2,
        )

        ax.add_patch(arrow)
    for node_id, node in genome.nodes.items():

        if node_id not in positions:
            continue

        x, y = positions[node_id]

        if node.type.name == "INPUT":
            node_color = "#4f8ef7"
            label = f"I{node_id}"
            radius = 0.25

        elif node.type.name == "BIAS":
            node_color = "#9b59b6"
            label = "B"
            radius = 0.25

        elif node.type.name == "HIDDEN":
            node_color = "#f5a623"
            label = f"H{node_id}"
            radius = 0.25

        else:
            node_color = "#35b66f"
            label = f"O{node_id}"
            radius = 0.28

        circle = Circle(
            (x, y),
            radius,
            facecolor=node_color,
            edgecolor="white",
            linewidth=2.5,
            zorder=5,
        )

        ax.add_patch(circle)

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=6,
        )

    for layer, node_ids in nodes_by_layer.items():

        x = layer * x_spacing

        if layer == 0:
            label = "INPUTS"

        elif any(
            genome.nodes[node_id].type.name == "OUTPUT"
            for node_id in node_ids
        ):
            label = "OUTPUT"

        else:
            label = f"HIDDEN {layer}"

        highest_y = max(
            positions[node_id][1]
            for node_id in node_ids
        )

        ax.text(
            x,
            highest_y + 1.0,
            label,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#374151",
        )

    scalar_mappable = ScalarMappable(
        norm=norm,
        cmap=cmap,
    )

    scalar_mappable.set_array([])

    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        shrink=0.75,
        pad=0.02,
    )

    colorbar.set_label(
        "Connection weight",
        fontsize=10,
    )

    ax.set_title(
        title,
        fontsize=18,
        fontweight="bold",
        color="#1f2937",
        pad=20,
    )

    legend_items = [
        ("Inputs", "#4f8ef7"),
        ("Bias", "#9b59b6"),
        ("Hidden", "#f5a623"),
        ("Output", "#35b66f"),
    ]

    for label, color in legend_items:
        ax.scatter(
            [],
            [],
            s=110,
            color=color,
            label=label,
        )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    margin = 1.5

    x_values = [
        position[0]
        for position in positions.values()
    ]

    y_values = [
        position[1]
        for position in positions.values()
    ]

    if x_values:
        ax.set_xlim(
            min(x_values) - margin,
            max(x_values) + margin,
        )

    if y_values:
        ax.set_ylim(
            min(y_values) - margin,
            max(y_values) + margin,
        )

    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_history(
    history,
    title="NEAT Training Progress",
    save_path=None,
    show=True,
):
    if not history:
        raise ValueError("Training history is empty.")

    history = np.asarray(history, dtype=float)
    generations = np.arange(1, len(history) + 1)

    # Best fitness seen up to each generation.
    best_so_far = np.maximum.accumulate(history)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    fig.patch.set_facecolor("#f7f8fc")
    ax.set_facecolor("#f7f8fc")

    # Fitness of the best genome in each generation
    ax.plot(
        generations,
        history,
        linewidth=2,
        alpha=0.5,
        label="Generation best",
    )

    # Best fitness discovered so far
    ax.plot(
        generations,
        best_so_far,
        linewidth=3,
        label="Best so far",
    )

    # Highlight final best
    ax.scatter(
        generations[-1],
        best_so_far[-1],
        s=80,
        zorder=5,
    )

    ax.annotate(
        f"{best_so_far[-1]:.2f}",
        xy=(
            generations[-1],
            best_so_far[-1],
        ),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(
        title,
        fontsize=17,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Generation",
        fontsize=11,
    )

    ax.set_ylabel(
        "Fitness",
        fontsize=11,
    )

    ax.grid(
        alpha=0.2,
        linestyle="--",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        loc="best",
    )

    # Show integer generations on x-axis
    ax.set_xticks(generations)

    # Don't let labels get unnecessarily cramped
    if len(generations) > 20:
        ax.set_xticks(
            np.linspace(
                1,
                len(generations),
                min(10, len(generations)),
                dtype=int,
            )
        )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    if show:
        plt.show()
    else:
        plt.close(fig)
