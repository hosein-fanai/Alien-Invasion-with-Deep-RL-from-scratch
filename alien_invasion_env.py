import pygame

import cv2

import numpy as np

from matplotlib import pyplot as plt

import random

from alien_invasion import AlienInvasion


class AlienInvasionEnv(AlienInvasion):

    def __init__(self, mode="human"):
        # super().__init__()
        # self.game_active = True
        # self.game_over = False
        # self._game_driver = self._run_game()

        # self.prev_info = {
        #     "ship left": self.stats.ship_left,
        #     "level": self.stats.level,
        #     "score": self.stats.score,
        #     "high_score": self.stats.high_score,
        #     "#activate bullets": len(self.bullets.sprites()),
        #     "#remaining aliens": len(self.aliens.sprites()),
        #     "alien point multiplier": self.settings.alien_points,
        # }

        self.action = 0
        self.mode = mode

    def _run_game(self):
        while True:
            self._check_inputs()

            self.ship.update()
            self._update_bullets()
            self._update_aliens()

            self._update_screen()

            yield self.render(self.mode)

    def _check_inputs(self):
        if self.action == 0: # NOOP
            pass
        elif self.action == 1: # RIGHT
            self.ship.moving_right = True
            self.ship.moving_left = False
        elif self.action == 2: # LEFT
            self.ship.moving_left = True
            self.ship.moving_right = False
        elif self.action == 3: # FIRE
            self._fire_bullet()
        elif self.action == 4: # LEFTFIRE
            self.ship.moving_left = True
            self.ship.moving_right = False
            self._fire_bullet()
        elif self.action == 5: # RIGHTFIRE
            self.ship.moving_right = True
            self.ship.moving_left = False
            self._fire_bullet()

    def _ship_hit(self):
        if self.stats.ship_left > 0:
            self.stats.ship_left -= 1
            self.sb.prep_ships()

            self.bullets.empty()
            self.aliens.empty()

            self._create_fleet()
            self.ship.center_ship()

            # time.sleep(0.5)
        else:
            self.game_active = False
            self.game_over = True
            pygame.mouse.set_visible(True)

    def _load_highscore(self):
        pass

    def _save_highscore(self):
        pass

    def _preprocess_obs(self, obs): # this is not optimized
        obs = np.mean(obs, axis=-1)
        obs = np.round(obs).astype(np.uint8)
        obs = np.transpose(obs, (1, 0))
        # obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[50:, :, np.newaxis]

        height, width, _ = obs.shape
        new_width = width // 4
        new_height = height // 4
        obs = cv2.resize(obs, (new_width, new_height), interpolation=cv2.INTER_LINEAR)[..., np.newaxis]

        return obs

    def _reward_func(self):
        alien_hits = (self.stats.score - self.prev_info["score"]) / self.prev_info["alien point multiplier"]
        ship_hit = self.stats.ship_left - self.prev_info["ship left"]
        level_up = self.stats.level - self.prev_info["level"]

        reward = 0

        if self.action == 0:
            reward -= 1
        elif self.action in (3, 4, 5):
            reward -= 2

        if ship_hit:
            reward -= 10
        else:
            reward += 1

        if alien_hits:
            reward += 5 * alien_hits

        if level_up:
            reward += 10

        return reward

    def step(self, action, mode="human"):
        self.action = action
        self.mode = mode
        state = next(self._game_driver)

        reward = self._reward_func()
        info = {
            "ship left": self.stats.ship_left,
            "level": self.stats.level,
            "score": self.stats.score,
            "high_score": self.stats.high_score,
            "#activate bullets": len(self.bullets.sprites()),
            "#remaining aliens": len(self.aliens.sprites()),
            "alien point multiplier": self.settings.alien_points,
        }
        done = self.game_over

        if done:
            self._restart_game()

        self.prev_info = info

        return state, reward, info, done

    def render(self, mode="human"):
        state = pygame.surfarray.array3d(self.screen)
        state = self._preprocess_obs(state)

        if mode == "rgb_array":
            pygame.display.iconify()
        elif mode == "human": # TODO fix this
            # if self.screen is None:
                # icon = pygame.image.load("ai.ico")
                # pygame.display.set_icon(icon)

                # self.screen = pygame.display.set_mode((self.settings.screen_dims[0], self.settings.screen_dims[1]))
                # self.screen_rect = self.screen.get_rect()

                # pygame.display.set_caption("Alien Invasion")
            pass

        return state

    def reset(self, mode="human"):
        self.close()

        super().__init__()
        self.game_active = True
        self.game_over = False
        self._game_driver = self._run_game()

        self.prev_info = {
            "ship left": self.stats.ship_left,
            "level": self.stats.level,
            "score": self.stats.score,
            "high_score": self.stats.high_score,
            "#activate bullets": len(self.bullets.sprites()),
            "#remaining aliens": len(self.aliens.sprites()),
            "alien point multiplier": self.settings.alien_points,
        }

        return self.render(mode=mode)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = AlienInvasionEnv()
    obs = env.reset()

    while True:
        i = random.randint(0, 5)
        obs, reward, info, done = env.step(i, mode="human")
        print(reward)

        # plt.imsave("last_frame.png", obs)

        # plt.imshow(obs, cmap='gray')
        # plt.show()

        # break
        pass