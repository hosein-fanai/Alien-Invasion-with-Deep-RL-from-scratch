from gym import Env
from gym.spaces import Box, Discrete

from alien_invasion_env import AlienInvasionEnv


class AlienInvasionGymEnv(Env):
    
    def __init__(self):
        self.game = AlienInvasionEnv()

        self.observation_space = Box(low=0, high=255, shape=(167, 320, 1), dtype="uint8")
        self.action_space = Discrete(6)

    def step(self, action, mode="human"):
        return self.game.step(action, mode)

    def render(self, mode="human"):
        return self.game.render()

    def reset(self, mode="human"):
        return self.game.reset(mode)

    def close(self):
        self.game.close()


if __name__ == "__main__":
    env = AlienInvasionGymEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, info, done = env.step(action)

    env.close()