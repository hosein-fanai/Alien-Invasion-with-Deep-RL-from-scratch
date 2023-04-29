class Settings:

    def __init__(self):
        self.screen_dims = 1280, 720 #854, 480
        self.bg_color = (230, 230, 230)

        self.ship_speed = 2.5
        self.ship_limit = 3

        self.bullet_speed = 2.5
        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = (60, 60, 60)
        self.bullets_allowed = 3

        self.alien_speed = 1.0
        self.fleet_drop_speed = 10
        self.fleet_direction = 1