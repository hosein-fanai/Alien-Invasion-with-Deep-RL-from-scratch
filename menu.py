import pygame

from button import Button


class Menu:

    def __init__(self, ai_game, button_msg):
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen_rect
        self.settings = ai_game.settings

        self.rect = pygame.Rect(0, 0, 500, 500)
        self.rect.center = self.screen_rect.center

        self.button = Button(ai_game, button_msg)

        self.font = pygame.font.SysFont(None, 40)
        self._render_guides()

    def draw(self):
        pygame.draw.rect(self.screen, self.settings.menu_bg_color, self.rect)
        pygame.draw.rect(self.screen, (0, 135, 0), self.rect, width=10)
        pygame.draw.rect(self.screen, (0, 0, 0), self.rect, width=3)

        self.button.draw_button()

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