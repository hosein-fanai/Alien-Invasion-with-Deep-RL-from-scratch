import pygame

import sys
import time

from settings import Settings
from game_stats import GameStats
from scoreboard import Scoreboard
from menu import Menu
from ship import Ship
from bullet import Bullet
from alien import Alien


class AlienInvasion:

    def __init__(self):
        pygame.init()

        self.game_active = False
        self.game_over = False
        self.blurred_screen = None

        self.settings = Settings()
        self.screen = pygame.display.set_mode(
            self.settings.screen_dims,
            # (0, 0), 
            # flags=pygame.FULLSCREEN
        )
        self.screen_rect = self.screen.get_rect()
        self.settings.screen_dims = self.screen_rect.width, self.screen_rect.height

        self.clock = pygame.time.Clock()
        self.ship = Ship(self)
        self.bullets = pygame.sprite.Group()
        self.aliens = pygame.sprite.Group()
        self.stats = GameStats(self)
        self.sb = Scoreboard(self)
        self.pause_menu = Menu(self, "Play")

        self._create_fleet()

        pygame.display.set_caption("Alien Invasion")

    def run_game(self):
        while True:
            self._check_events()

            if self.game_active:
                self.ship.update()
                self._update_bullets()
                self._update_aliens()

            self._update_screen()            
            self.clock.tick(60)

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                self._check_keydown_events(event)
            elif event.type == pygame.KEYUP:
                self._check_keyup_events(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pose = pygame.mouse.get_pos()
                self._check_play_button(mouse_pose)

    def _check_keydown_events(self, event):
        if event.key == pygame.K_RIGHT:
            self.ship.moving_right = True
        elif event.key == pygame.K_LEFT:
            self.ship.moving_left = True
        elif event.key == pygame.K_q:
                sys.exit()
        elif event.key == pygame.K_SPACE:
            if not self.game_active:
                self._resume_game()
                return
            self._fire_bullet()
        elif event.key == pygame.K_ESCAPE:
            self._pause_game()
        elif event.key == pygame.K_p:
            if not self.game_active:
                self._resume_game()

    def _check_keyup_events(self, event):
        if event.key == pygame.K_RIGHT:
            self.ship.moving_right = False
        elif event.key == pygame.K_LEFT:
            self.ship.moving_left = False

    def _check_play_button(self, mouse_pose):
        button_clicked = self.pause_menu.button.rect.collidepoint(mouse_pose)
        if button_clicked and not self.game_active:
            if self.game_over:
                self._restart_game()
            else:
                self._resume_game()

    def _restart_game(self):
        self.game_active = True
        self.game_over = False
        self.blurred_screen = None

        self.settings.initialize_dynamic_settings()
        self.stats.reset_stats()
        self.sb.prep_score()
        self.sb.prep_level()
        self.sb.prep_ships()

        self.bullets.empty()
        self.aliens.empty()

        self._create_fleet()
        self.ship.center_ship()

        pygame.mouse.set_visible(False)

    def _resume_game(self):
        self.blurred_screen = None
        self.game_active = True
        pygame.mouse.set_visible(False)

    def _pause_game(self):
        self.game_active = False
        pygame.mouse.set_visible(True)

    def _blur_fg(self):
        if not self.blurred_screen:
            self._save_blurred_screen()

        self.screen.blit(self.blurred_screen, self.screen_rect)

    def _save_blurred_screen(self):
        blur_sf = pygame.Surface(self.screen.get_size())
        blur_sf.set_alpha(210)
        blur_sf.fill((35, 35, 35))
        self.screen.blit(blur_sf, (0, 0))

        scale = 1 / 4
        surf_size = self.screen.get_size()
        scale_size = (int(surf_size[0]*scale), int(surf_size[1]*scale))
        surf = pygame.transform.smoothscale(self.screen, scale_size)
        surf = pygame.transform.smoothscale(surf, surf_size)

        self.blurred_screen = surf

    def _update_screen(self):
        self.screen.fill(self.settings.bg_color)

        for bullet in self.bullets.sprites():
            bullet.draw_bullet()

        self.ship.blitme()

        self.aliens.draw(self.screen)

        self.sb.draw_score()

        if not self.game_active:
            self._blur_fg()
            self.pause_menu.draw()

        pygame.display.flip()

    def _fire_bullet(self):
        if len(self.bullets) < self.settings.bullets_allowed:
            self.bullets.add(Bullet(self))

    def _update_bullets(self):
        self.bullets.update()
        for bullet in self.bullets.copy():
            if bullet.rect.bottom < 0:
                self.bullets.remove(bullet)

        self._check_bullet_alien_collision()

    def _create_fleet(self):
        alien = Alien(self)
        alien_width, alien_height = alien.rect.size

        current_x, current_y = alien_width, alien_height
        while current_y < (self.settings.screen_dims[1] - 3 * alien_height):
            while current_x < (self.settings.screen_dims[0] - 2 * alien_width):
                self._create_alien(current_x, current_y)
                current_x += 2 * alien_width

            current_x = alien_width
            current_y += 2 * alien_height

    def _create_alien(self, current_x, current_y):
        alien = Alien(self)
        alien.x = current_x
        alien.rect.x = current_x
        alien.rect.y = current_y
        self.aliens.add(alien)

    def _update_aliens(self):
        self._check_fleet_edges()
        self.aliens.update()

        if pygame.sprite.spritecollideany(self.ship, self.aliens):
            self._ship_hit()

        self._check_aliens_bottom()

    def _check_fleet_edges(self):
        for alien in self.aliens.sprites():
            if alien.check_edges():
                self._change_fleet_direction()
                break

    def _change_fleet_direction(self):
        for alien in self.aliens.sprites():
            alien.rect.y += self.settings.fleet_drop_speed
        self.settings.fleet_direction *= -1

    def _check_bullet_alien_collision(self):
        collisions = pygame.sprite.groupcollide(self.bullets, self.aliens, True, True)

        if collisions:
            for alien in collisions.values():
                self.stats.score += self.settings.alien_points * len(alien)
            self.sb.prep_score()
            self.sb.check_hight_score()

        if not self.aliens:
            self.bullets.empty()
            self._create_fleet()

            self.stats.level += 1
            self.sb.prep_level()

            self.settings.increase_speed()

    def _ship_hit(self):
        if self.stats.ship_left > 0:
            self.stats.ship_left -= 1
            self.sb.prep_ships()

            self.bullets.empty()
            self.aliens.empty()

            self._create_fleet()
            self.ship.center_ship()

            time.sleep(0.5)
        else:
            self.game_active = False
            self.game_over = True
            pygame.mouse.set_visible(True)

    def _check_aliens_bottom(self):
        for alien in self.aliens.sprites():
            if alien.rect.bottom >= self.settings.screen_dims[1]:
                self._ship_hit()
                break


if __name__ == "__main__":
    ai_game = AlienInvasion()
    ai_game.run_game()