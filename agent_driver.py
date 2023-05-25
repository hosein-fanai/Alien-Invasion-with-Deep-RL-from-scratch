import tensorflow as tf
from tensorflow.keras import models

from utils import make_env

import sys

import time


if __name__ == "__name__":
    try:
        model_path = sys.argv[0]
    except:
        model_path = "models/old/Alien Invasion DQN (100 iters).h5"

    model = models.load_model(model_path)

    env = make_env()

    obs = env.reset()
    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        action = tf.argmax(model(obs[None]), axis=1)
        obs, reward, done, info = env.step(action)
        rewards += reward

        env.game.clock.tick(60)
        frames += 1
    env.close()

    fps = int(frames // (time.time() - start))
    print(f"Rewards: {rewards}, FPS:{fps}")
