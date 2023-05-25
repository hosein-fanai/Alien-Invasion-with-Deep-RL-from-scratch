from tensorflow.keras import models

from utils import make_env, DQNAgent

import sys

import time


if __name__ == "__main__":

    try:
        model_path = sys.argv[1]

        if len(sys.argv) == 3:
            fps = int(sys.argv[2])
        else:
            fps = 60
    except:
        model_path = "models/Alien Invasion DQN (425%500 iters & 0.1568 epsilon).h5"

    env = make_env(game_resolution=(720, 720))
    obs = env.reset()

    model = models.load_model(model_path)
    agent = DQNAgent(env, model, None, None, None, None)
    
    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        action = agent.epsilon_greedy_policy(obs, 0.1568)
        obs, reward, done, info = env.step(action)
        rewards += reward

        env.game.clock.tick(fps)
        frames += 1
    env.close()

    fps = int(frames // (time.time() - start))
    print(f"Rewards: {rewards}, FPS:{fps}")