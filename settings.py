class Settings:

    def __init__(self):
        self.screen_dims = 1280, 720 #854, 480
        self.bg_color = (230, 230, 230)
        self.menu_bg_color = (198, 255, 191)
        
        self.ship_limit = 3
        
        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)
        self.bullets_allowed = 3
        
        self.fleet_drop_speed = 10

        self.speed_scale = 1.1
        self.score_scale = 1.5

        self.initialize_dynamic_settings()

    def initialize_dynamic_settings(self):
        self.ship_speed = 2.5 #1.5

        self.bullet_speed = 2.5

        self.alien_speed = 1.0
        self.alien_points = 50
        self.fleet_direction = 1

    def increase_speed(self):
        self.ship_speed *= self.speed_scale
        self.bullet_speed *= self.speed_scale
        self.alien_speed *= self.speed_scale

        self.alien_points = int(self.alien_points * self.score_scale)

    def decrement_speed(self):
        self.ship_speed /= self.speed_scale
        self.bullet_speed /= self.speed_scale
        self.alien_speed /= self.speed_scale

        self.alien_points = int(self.alien_points / self.score_scale)