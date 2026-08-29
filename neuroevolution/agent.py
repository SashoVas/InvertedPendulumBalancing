import torch
from torch import nn
import pygame

INPUT_LAYER_SIZE = 5
HIDDEN_LAYER_SIZE = 5
OUTPUT_LAYER_SIZE = 3


class AgentNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.neural_network = nn.Sequential(
            nn.Linear(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE),
        )

    def forward(self, x):
        logits = self.neural_network(x)
        return logits


class Agent:
    def __init__(self, state_dict=None):
        self.model = AgentNeuralNetwork()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

    def update_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        state_dict = torch.load(file_path)
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def get_action(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_logits = self.model(state_tensor)
            action = torch.argmax(action_logits).item()

            if action == 0:
                action = {pygame.K_RIGHT: False, pygame.K_LEFT: False}
            elif action == 1:
                action = {pygame.K_RIGHT: True, pygame.K_LEFT: False}
            else:
                action = {pygame.K_RIGHT: False, pygame.K_LEFT: True}

            return action
