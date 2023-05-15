import pygame


class Button:

    def __init__(self, ai_game, msg):
        self.screen = ai_game.screen
        self.screen_rect = self.screen.get_rect()

        self.width, self.height = 200, 50
        self.default_button_color = (0, 135, 0)
        self.hovered_button_color = (20, 180, 20)
        self.text_color = (255, 255, 255)
        self.font = pygame.font.SysFont(None, 48)

        self.rect = pygame.Rect(0, 0, self.width, self.height)
        self.rect.center = self.screen_rect.center

        self._prep_msg(msg)

    def _prep_msg(self, msg, hovered=False):
        self.last_msg = msg
        self.current_color = self.hovered_button_color if hovered else self.default_button_color
        self.msg_image = self.font.render(msg, True, self.text_color, self.current_color)
        self.msg_image_rect = self.msg_image.get_rect()

        self.msg_image_rect.center = self.rect.center

    def draw_button(self):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            self._prep_msg(self.last_msg, True)
        elif self.current_color != self.default_button_color:
            self._prep_msg(self.last_msg, False)
        self.screen.fill(self.current_color, self.rect)
        
        self.screen.blit(self.msg_image, self.msg_image_rect)

        pygame.draw.rect(self.screen, (0, 0, 0), self.rect, width=3)