from alien_invasion import AlienInvasion
from alien import Alien

import pygame

import cv2

import numpy as np

# from matplotlib import pyplot as plt

import random

import time


class AlienInvasionEnv(AlienInvasion):

    def __init__(self, preprocess_obs=False, render_type="gray_scale", **kwargs):
        super().__init__(**kwargs)
        self.game_active = True
        self.game_over = False

        # self.mode = mode
        self.preprocess_obs = preprocess_obs
        self.render_type = render_type

        # self._game_driver = self._run_game()
        self.action = 0
        self.prev_info = self._get_info

    def _run_game(self):
        while True:
            self._check_inputs()

            self.ship.update()
            self._update_bullets()
            self._update_aliens()

            self._update_screen()

            yield self.render(mode=self.mode)

    def _run_game2(self):
        self._check_inputs()

        self.ship.update()
        self._update_bullets()
        self._update_aliens()

        self._update_screen()

        return self.render(self.mode)

    def _check_inputs(self):
        if self.action == 0: # NOOP
            self.ship.moving_left = False
            self.ship.moving_right = False
        elif self.action == 1: # LEFT
            self.ship.moving_right = False

            self.ship.moving_left = True
        elif self.action == 2: # RIGHT
            self.ship.moving_left = False
            
            self.ship.moving_right = True
        elif self.action == 3: # FIRE
            self.ship.moving_left = False
            self.ship.moving_right = False

            self._fire_bullet()
        elif self.action == 4: # LEFTFIRE
            self.ship.moving_right = False

            self.ship.moving_left = True
            self._fire_bullet()
        elif self.action == 5: # RIGHTFIRE
            self.ship.moving_left = False

            self.ship.moving_right = True
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

    def _create_alien(self, current_x, current_y, row, col):
        alien = Alien(self, (row, col))
        alien.x = current_x
        alien.rect.x = current_x
        alien.rect.y = current_y
        self.aliens.add(alien)

    def _create_fleet(self):
        self.fleet_state = np.ones((5, 10), dtype=np.uint8)

        alien = Alien(self)
        alien_width, alien_height = alien.rect.size

        current_x, current_y = alien_width, alien_height
        row = 0
        while current_y < (self.settings.screen_dims[1] - 3 * alien_height):
            col = 0
            while current_x < (self.settings.screen_dims[0] - 2 * alien_width):
                self._create_alien(current_x, current_y, row, col)
                current_x += 2 * alien_width

                col += 1

            current_x = alien_width
            current_y += 2 * alien_height

    def _check_bullet_alien_collision(self):
        collisions = pygame.sprite.groupcollide(self.bullets, self.aliens, True, True)

        if collisions:
            for alien in collisions.values():
                self.stats.score += self.settings.alien_points * len(alien)
                for a in alien:
                    alien_row, alien_col = a.label
                    self.fleet_state[alien_row][alien_col] = 0

            self.sb.prep_score()
            self.sb.check_hight_score()

        if not self.aliens:
            self.bullets.empty()
            self._create_fleet()

            self._increment_level()

    def _load_highscore(self):
        pass

    def _save_highscore(self):
        pass

    def _preprocess_obs(self, obs):
        if self.preprocess_obs:
            # obs = np.mean(obs, axis=-1)
            # obs = np.round(obs).astype(np.uint8)
            # obs = obs[..., np.newaxis]

            obs = np.transpose(obs, (1, 0, 2))
            obs = obs[50:, :]

            height, width, _ = obs.shape
            new_width = width // 4
            new_height = height // 4
            obs = cv2.resize(obs, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            obs = obs[..., np.newaxis]

        return obs

    def _reward_func(self, info):
        ship_distance_from_center = abs(self.ship.rect.centerx - self.screen_rect.centerx)
        remaining_aliens = info["#remaining aliens"]
        alien_hits = self.prev_info["#remaining aliens"] - info["#remaining aliens"]
        bullet_wasted = self.prev_info["#active bullets"] - info["#active bullets"]
        ship_hit = info["ship left"] - self.prev_info["ship left"]
        level_up = info["level"] - self.prev_info["level"]

        reward = 0

        # reward -= ship_distance_from_center / 1000

        reward -= remaining_aliens / 500

        # if self.action == 0:
        #     reward -= 1
        # elif self.action in (3, 4, 5):
        #     reward -= 2

        if bullet_wasted and not alien_hits:
            reward -= 5 * bullet_wasted

        if ship_hit:
            reward -= 500
        # else:
        #     reward += 0.07

        if alien_hits: # needs to be optimized
            reward += 10 * alien_hits

            for prev_row, row in zip(self.prev_info["remaining fleet row"], info["remaining fleet row"]):
                if prev_row != row:
                    reward += 100
            for prev_col, col in zip(self.prev_info["remaining fleet col"], info["remaining fleet col"]):
                if prev_col != col:
                    reward += 50

        if level_up:
            reward += 300

        return reward

    def _get_info(self):
        fleet_row_state = (self.fleet_state @ np.zeros((10, 1))).flatten()
        fleet_col_state = (self.fleet_state.T @ np.zeros((5, 1))).flatten()

        return {
            "ship left": self.stats.ship_left,
            "level": self.stats.level,
            "score": self.stats.score,
            "high_score": self.stats.high_score,
            "#active bullets": len(self.bullets.sprites()),
            "#remaining aliens": len(self.aliens.sprites()),
            "remaining fleet row": fleet_row_state,
            "remaining fleet col": fleet_col_state,
            "alien point multiplier": self.settings.alien_points,
        }

    def step(self, action, mode="human"):
        self.action = action
        self.mode = mode
        # obs = next(self._game_driver)
        obs = self._run_game2()

        info = self._get_info()
        reward = self._reward_func(info)
        done = self.game_over

        if done:
            self._restart_game()

        self.prev_info = info

        return obs, reward, done, info

    def render(self, mode="human"): # Needs furthur optimization
        pixel_array = pygame.PixelArray(self.screen)
        if self.render_type == "gray_scale":
            obs = np.array(pixel_array, dtype="uint8")[..., np.newaxis]
            del pixel_array
        elif self.render_type == "rgb":
            pixel_array = np.array(pixel_array)
            obs = np.zeros((self.screen.get_width(), self.screen.get_height(), 3), dtype=np.uint8)
            obs[:, :, 0] = (pixel_array >> 16) & 0xFF  # Red channel
            obs[:, :, 1] = (pixel_array >> 8) & 0xFF   # Green channel
            obs[:, :, 2] = pixel_array & 0xFF          # Blue channel

        obs = self._preprocess_obs(obs)

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

        return obs

    def reset(self, mode="human"):
        # self.close()

        # self._game_driver = self._run_game()
        self._restart_game()
        self.action = 0
        self.prev_info = self._get_info()

        return self.render(mode=mode)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = AlienInvasionEnv(
        # game_resolution=(720, 720), 
        preprocess_obs=False
    )

    obs = env.reset()

    done = False
    rewards = 0

    frames = 0
    start = time.time()
    while not done:
        i = random.randint(0, 5)
        obs, reward, done, info = env.step(i, mode="human")
        rewards += reward

        print(f"\r{rewards}", end="")
        # print(reward)

        # plt.imsave("last_frame.png", obs)

        # plt.imshow(obs, cmap='gray')
        # plt.show()
    
        frames += 1
    print()
    print(frames / (time.time() - start))