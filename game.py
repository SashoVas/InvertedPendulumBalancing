import sys

import numpy as np
import pygame
import torch
from constants import *
from pendulums import SinglePendulum, DoublePendulum
from neuroevolution.agent import Agent
from neat.neat import NeatAgent


def draw_text(screen, font, text, pos, color=TEXT_COLOR):
    screen.blit(font.render(text, True, color), pos)


def draw_scene(screen, font, pendulum, score, best_score, time_left, round_over):
    screen.fill(BG_COLOR)

    pygame.draw.line(
        screen, TRACK_COLOR,
        (SCREEN_W / 2 - TRACK_HALF_WIDTH, TRACK_Y),
        (SCREEN_W / 2 + TRACK_HALF_WIDTH, TRACK_Y),
        4,
    )

    points = pendulum.joint_positions()

    cart_rect = pygame.Rect(0, 0, 60, 30)
    cart_rect.center = points[0]
    pygame.draw.rect(screen, CART_COLOR, cart_rect, border_radius=4)

    for i in range(1, len(points)):
        color = LINK_COLORS[i - 1]
        pygame.draw.line(screen, color, points[i - 1], points[i], 5)
        pygame.draw.circle(
            screen, color, (int(points[i][0]), int(points[i][1])), 12)

    draw_text(screen, font, f"Mode: {pendulum.name}", (20, 20))
    draw_text(screen, font, f"Score: {score:.1f}", (20, 50))
    draw_text(screen, font, f"Best: {best_score:.1f}", (20, 80))
    draw_text(screen, font, f"Time left: {time_left:.1f}s", (20, 110))
    draw_text(screen, font, "Arrows: push cart   M: switch mode   R: reset",
              (20, SCREEN_H - 40))

    if round_over:
        draw_text(screen, font, f"Time's up! Final score: {score:.1f}  (Press R to play again)",
                  (SCREEN_W / 2 - 280, SCREEN_H / 2 - 120), HIGHLIGHT_COLOR)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def play_game(render=True, agent=None):

    if render:
        pygame.init()
        screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Pendulum Balance")
        font = pygame.font.SysFont("consolas", 22)
        clock = pygame.time.Clock()

    modes = {"single": SinglePendulum(), "double": DoublePendulum()}
    mode_names = list(modes.keys())
    mode_index = 0
    pendulum = modes[mode_names[mode_index]]

    best_scores = {name: 0.0 for name in modes}

    def start_new_round():
        pendulum.reset()
        return 0.0, 0.0, False  # elapsed, score, round_over

    elapsed, score, round_over = start_new_round()
    fitness = 0.0
    score_timer = 0.0  # accumulates dt until the next scoring tick

    running = True
    while running:
        if render:
            dt = min(clock.tick(FPS) / 1000.0, MAX_DT)
        else:
            dt = 1.0 / FPS

        if render and agent is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        elapsed, score, round_over = start_new_round()
                        score_timer = 0.0
                    elif event.key == pygame.K_m:
                        mode_index = (mode_index + 1) % len(mode_names)
                        pendulum = modes[mode_names[mode_index]]
                        elapsed, score, round_over = start_new_round()
                        score_timer = 0.0

            keys = pygame.key.get_pressed()
        else:

            keys = {}
            if agent is not None:
                state = [
                    pendulum.x / TRACK_LIMIT,
                    pendulum.x_dot / X_DOT_SCALE,
                    np.sin(pendulum.theta),
                    np.cos(pendulum.theta),
                    pendulum.theta_dot / THETA_DOT_SCALE,
                ]
                keys = agent.get_action(state)

        force = 0.0
        if keys[pygame.K_LEFT]:
            force -= FORCE_MAG
        if keys[pygame.K_RIGHT]:
            force += FORCE_MAG

        if not round_over:
            sub_dt = dt / SUBSTEPS
            for _ in range(SUBSTEPS):
                pendulum.step(force, sub_dt)

            elapsed += dt

            score_timer += dt

            upright = pendulum.upright_fraction()
            position_penalty = abs(pendulum.x) / TRACK_LIMIT
            velocity_penalty = abs(pendulum.x_dot) / X_DOT_SCALE
            angular_velocity_penalty = abs(
                pendulum.theta_dot
            ) / THETA_DOT_SCALE

            while score_timer >= SCORE_TICK:
                score_timer -= SCORE_TICK
                score += POINTS_PER_TICK * upright
                fitness += (
                    upright
                    - 0.2 * position_penalty
                    - 0.05 * velocity_penalty
                    - 0.05 * angular_velocity_penalty
                ) * dt

            if elapsed >= ROUND_DURATION:
                round_over = True
                current_mode = mode_names[mode_index]
                best_scores[current_mode] = max(
                    best_scores[current_mode], score)

        time_left = max(0.0, ROUND_DURATION - elapsed)
        if not render and round_over:
            break
        if render:
            draw_scene(screen, font, pendulum, score,
                       best_scores[mode_names[mode_index]], time_left, round_over)
            pygame.display.flip()

    if render:
        pygame.quit()
        sys.exit()

    return score, fitness


if __name__ == "__main__":
    # agent = Agent()
    # agent.load_state_dict(torch.load("agents/neuroevolution.pth"))
    agent = NeatAgent.load_from_file("agents/best_neat_genome.pkl")
    play_game(render=True, agent=agent)
