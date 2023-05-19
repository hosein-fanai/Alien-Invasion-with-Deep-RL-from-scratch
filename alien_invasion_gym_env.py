from gym import Env
from gym.spaces import Box, Discrete

from alien_invasion_env import AlienInvasionEnv


class AlienInvasionGymEnv(Env):
    
    def __init__(self, **kwargs):
        self.game = AlienInvasionEnv(**kwargs)

        shape = (167, 320, 1) if self.game.preprocess_obs else (1280, 720, 3)
        if self.game.render_type == "gray_scale":
            shape = (shape[0], shape[1], 1)

        self.observation_space = Box(low=0, high=255, shape=shape, dtype="uint8")
        self.action_space = Discrete(6)

    def step(self, action, mode="human"):
        return self.game.step(action, mode)

    def render(self, mode="human"):
        return self.game.render(mode)

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
        obs, reward, done, info = env.step(action)

    env.close()