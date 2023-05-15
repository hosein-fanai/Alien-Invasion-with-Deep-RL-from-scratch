import pygame

from button import Button


class Menu:

    def __init__(self, ai_game):
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen_rect
        self.settings = ai_game.settings

        self.rect = pygame.Rect(0, 0, 500, 500)
        self.rect.center = self.screen_rect.center

        self.play_button = Button(ai_game, "Play")

        self.font = pygame.font.SysFont(None, 40)
        self._render_guides()

        self.inc_level_button = Button(ai_game, "+")
        self.inc_level_button.rect.center = self.rect.midright
        self.inc_level_button.rect.y -= 190
        self.inc_level_button.rect.width = 50
        self.inc_level_button._prep_msg("+")

        self.prep_level(ai_game.stats.level)

        self.dec_level_button = Button(ai_game, "-")
        self.dec_level_button.rect.center = self.rect.midright
        self.dec_level_button.rect.y -= 90
        self.dec_level_button.rect.width = 50
        self.dec_level_button._prep_msg("-")

    def draw(self, game_over):
        pygame.draw.rect(self.screen, self.settings.menu_bg_color, self.rect)
        pygame.draw.rect(self.screen, (0, 135, 0), self.rect, width=10)
        pygame.draw.rect(self.screen, (0, 0, 0), self.rect, width=3)

        self.play_button.draw_button()

        if game_over:
            self.inc_level_button.draw_button()
            self.dec_level_button.draw_button()

            self.screen.blit(self.level_image, self.level_image_rect)

        for guide_image, guide_image_rect in self.guides:
            self.screen.blit(guide_image, guide_image_rect)

    def _render_guides(self):
        guides_msgs = [
            "Shoot : (SPACE)",
            "Left : (LEFT ARROW)",
            "Right : (RIGHT ARROW)",
            "Pause : (ESC)",
            "Resume : (P / SPACE / Click)",
            "Quit : (Q)"
        ]
        pos = self.rect.topleft
        guides_poses = [(pos[0]+50, pos[1]+i*50+(120 if i>3 else 0)) 
                        for i in range(1, len(guides_msgs)+1)]

        self.guides = []
        for guide_msg, guide_pos in zip(guides_msgs, guides_poses):
            guide_image = self.font.render(guide_msg, True, (0, 0, 0), self.settings.menu_bg_color)
            guide_image_rect = guide_image.get_rect()
            guide_image_rect.topleft = guide_pos
            
            self.guides.append((guide_image, guide_image_rect))

    def prep_level(self, level):
        font = pygame.font.SysFont(None, 32, italic=True)
        
        self.level_image = font.render(f"Level: {level}", True, (0, 0, 0), self.settings.menu_bg_color)
        self.level_image_rect = self.level_image.get_rect()
        self.level_image_rect.center = self.rect.midright
        self.level_image_rect.y -= 140
        self.level_image_rect.x -= 80