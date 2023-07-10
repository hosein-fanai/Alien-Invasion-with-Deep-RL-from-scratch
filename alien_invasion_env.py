from alien_invasion import AlienInvasion
from alien import Alien

import pygame

import cv2

import numpy as np

# from matplotlib import pyplot as plt

import random

import time


class AlienInvasionEnv(AlienInvasion):

    def __init__(self, preprocess_obs=False, render_type="gray_scale", reset_pygame_import=False, difficulty=None, **kwargs):
        super().__init__(**kwargs)
        self.game_active = True
        self.game_over = False

        self.manual_use = False

        # self.mode = mode
        self.preprocess_obs = preprocess_obs
        self.render_type = render_type
        self.reset_pygame_import = reset_pygame_import
        self.difficulty = difficulty

        # self._game_driver = self._run_game()
        self.action = 0

        self._restart_game()
        self.prev_info = self._get_info()

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
            if not self.manual_use:
                self.ship.moving_left = False
                self.ship.moving_right = False
        elif self.action == 1: # LEFT
            if not self.manual_use:
                self.ship.moving_right = False
                self.ship.moving_left = True
        elif self.action == 2: # RIGHT
            if not self.manual_use:
                self.ship.moving_left = False
                self.ship.moving_right = True
        elif self.action == 3: # FIRE
            if not self.manual_use:
                self.ship.moving_left = False
                self.ship.moving_right = False

            self._fire_bullet()
        elif self.action == 4: # LEFTFIRE
            if not self.manual_use:
                self.ship.moving_right = False
                self.ship.moving_left = True

            self._fire_bullet()
        elif self.action == 5: # RIGHTFIRE
            if not self.manual_use:
                self.ship.moving_left = False
                self.ship.moving_right = True

            self._fire_bullet()

    def _check_events(self):
        is_event_key_pressed = False
        for event in pygame.event.get([pygame.KEYDOWN, pygame.KEYUP]):
            if event.type == pygame.KEYDOWN:
                self._check_keydown_events(event)
                is_event_key_pressed = True
            elif event.type == pygame.KEYUP:
                self._check_keyup_events(event)
                is_event_key_pressed = True
            break

        if is_event_key_pressed:
            match event.key:
                case pygame.K_LEFT:
                    return 1
                case pygame.K_RIGHT:
                    return 2
                case pygame.K_SPACE:
                    if self.ship.moving_left == True:
                        return 4
                    elif self.ship.moving_right == True:
                        return 5
                    else:
                        return 3
        return 0

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

            row += 1

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

    def _restart_game(self):
        super()._restart_game()

        match self.difficulty:
            case 0:
                self.fleet_state[0, :] = 0
                self.fleet_state[:, (0, 1, 8, 9)] = 0
            case 1:
                self.fleet_state[:, (0, 1, 8, 9)] = 0
            case 2:
                self.fleet_state[:, (0, 8, 9)] = 0
            case 3:
                self.fleet_state[:, (0, 9)] = 0
            case 4:
                self.fleet_state[:, 9] = 0
            case _:
                pass

        for alien in self.aliens.sprites():
            if self.fleet_state[alien.label] == 0:
                self.aliens.remove(alien)

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
        # ship_distance_from_center = abs(self.ship.rect.centerx - self.screen_rect.centerx)
        # remaining_aliens = info["#remaining aliens"]
        alien_hits = self.prev_info["#remaining aliens"] - info["#remaining aliens"]
        bullet_wasted = self.prev_info["#active bullets"] - info["#active bullets"]
        ship_hit = info["ship left"] - self.prev_info["ship left"]
        level_up = info["level"] - self.prev_info["level"]

        reward = 0

        # reward -= ship_distance_from_center / 1000

        # reward -= remaining_aliens / 500

        # if self.action == 0:
        #     reward -= 1
        # elif self.action in (3, 4, 5):
        #     reward -= 2

        if bullet_wasted > 0 and not alien_hits and not level_up:
            reward -= 1 * bullet_wasted

        if ship_hit or self.game_over:
            reward -= 10
        # else:
        #     reward += 0.07

        if alien_hits and not ship_hit and not level_up: # needs to be optimized
            reward += 1 * alien_hits

            for prev_row, row in zip(self.prev_info["remaining fleet row"], info["remaining fleet row"]):
                if prev_row != row and row == 0:
                    reward += 10
            for prev_col, col in zip(self.prev_info["remaining fleet col"], info["remaining fleet col"]):
                if prev_col != col and col == 0:
                    reward += 5

        if level_up:
            reward += 20

        return reward

    def _get_info(self):
        fleet_row_state = (self.fleet_state @ np.ones((10, 1))).flatten()
        fleet_col_state = (self.fleet_state.T @ np.ones((5, 1))).flatten()

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

        if self.render_type == "gray_scale" or mode == "bw_aray":
            obs = np.array(pixel_array, dtype="uint8")[..., np.newaxis]
            obs = self._preprocess_obs(obs)
        if self.render_type == "rgb" or mode == "rgb_array":
            pixel_array = np.array(pixel_array)
            obs = np.zeros((self.screen.get_width(), self.screen.get_height(), 3), dtype=np.uint8)
            obs[:, :, 0] = (pixel_array >> 16) & 0xFF  # Red channel
            obs[:, :, 1] = (pixel_array >> 8) & 0xFF   # Green channel
            obs[:, :, 2] = pixel_array & 0xFF          # Blue channel
            obs = np.transpose(obs, (1, 0, 2))
        
        del pixel_array

        if mode == "rgb_array":
            # pygame.display.iconify()
            pass
        if mode == "bw_aray":
            # pygame.display.iconify()
            pass
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
        if self.reset_pygame_import:
            self.close()
            self.__init__(preprocess_obs=True)
        else:
            # self._game_driver = self._run_game()
            self._restart_game()
            self.action = 0
            self.prev_info = self._get_info()

        return self.render(mode=mode)

    def close(self):
        # pass
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