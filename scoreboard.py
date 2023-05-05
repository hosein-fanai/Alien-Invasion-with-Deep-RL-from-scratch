import pygame

from ship import Ship


class Scoreboard:

    def __init__(self, ai_game):
        self.screen = ai_game.screen
        self.screen_rect = self.screen.get_rect()
        self.settings = ai_game.settings
        self.stats = ai_game.stats

        self.text_color = (30, 30, 30)
        self.font = pygame.font.SysFont(None, 48)

        self.prep_score()
        self.prep_high_score()
        self.prep_level()
        self.prep_ships()

    def prep_score(self):
        rounded_score = round(self.stats.score, -1)
        self.score_image = self.font.render(f"{rounded_score: ,}", 
                                        True, self.text_color, 
                                        self.settings.bg_color)

        self.score_image_rect = self.score_image.get_rect()
        self.score_image_rect.right = self.screen_rect.right - 20

    def prep_high_score(self):
        rounded_high_score = round(self.stats.high_score, -1)
        self.high_score_image = self.font.render(f"{rounded_high_score: ,}", 
                                        True, self.text_color, 
                                        self.settings.bg_color)

        self.high_score_image_rect = self.high_score_image.get_rect()
        self.high_score_image_rect.midtop = self.screen_rect.midtop

    def prep_level(self):
        level_str = str(self.stats.level)
        self.level_image = self.font.render(level_str, 
                                        True, self.text_color, 
                                        self.settings.bg_color)

        self.level_image_rect = self.level_image.get_rect()
        self.level_image_rect.right = self.screen_rect.right
        self.level_image_rect.top = self.score_image_rect.bottom + 10

    def prep_ships(self):
        self.ships = pygame.sprite.Group()
        for ship_number in range(self.stats.ship_left):
            ship = Ship(self)
            ship.rect.x = 10 + ship_number * ship.rect.width
            ship.rect.y = 10
            self.ships.add(ship)

    def check_hight_score(self):
        if self.stats.score > self.stats.high_score:
            self.stats.high_score = self.stats.score
            self.prep_high_score()

    def draw_score(self):
        self.screen.blit(self.score_image, self.score_image_rect)
        self.screen.blit(self.high_score_image, self.high_score_image_rect)
        self.screen.blit(self.level_image, self.level_image_rect)
        self.ships.draw(self.screen)