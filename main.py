import pygame
import random

WIDTH = 768
HEIGHT = 672
FPS = 30

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

IS_RESTART = True
IS_VICTORY = True
TIME = 0

NAME = "Fantastic game"
ICON_PATH = 'assets/soldier.png'

START_IMG_PATH = 'assets/start.png'
VICTORY_IMG_PATH = 'assets/victory.jpg'
DEFEAT_IMG_PATH = 'assets/defeat.png'

PLAYER_DOWN_IMG_PATH = 'assets/tank-player-down.png'
PLAYER_UP_IMG_PATH = 'assets/tank-player-up.png'
PLAYER_LEFT_IMG_PATH = 'assets/tank-player-left.png'
PLAYER_RIGHT_IMG_PATH = 'assets/tank-player-right.png'

ENEMY_DOWN_IMG_PATH = 'assets/tank-enemy-down.png'
ENEMY_UP_IMG_PATH = 'assets/tank-enemy-up.png'
ENEMY_LEFT_IMG_PATH = 'assets/tank-enemy-left.png'
ENEMY_RIGHT_IMG_PATH = 'assets/tank-enemy-right.png'

BRICK_IMG_PATH = 'assets/brick.png'
BULLET_IMG_PATH = 'assets/bullet.png'
GRASS_IMG_PATH = 'assets/grass.jpg'
HEART_IMG_PATH = 'assets/heart.jpg'

PLAYER_COORDINATES = (350, 570)
ENEMY_COORDINATES = [(50, 50, 'down'),
                     (WIDTH - 60, 60, 'up'),
                     (55, 400, 'right'),
                     (60, 560, 'left'),
                     (550, 350, 'down')]
BRICK_COORDINATES = list()
HEART_COORDINATES = [(WIDTH - 60, 50),
                     (WIDTH - 100, 50),
                     (WIDTH - 140, 50)]

SPRITES = pygame.sprite.Group()
ENEMIES = pygame.sprite.Group()
BRICKS = pygame.sprite.Group()
PLAYER_BULLETS = pygame.sprite.Group()
ENEMY_BULLETS = pygame.sprite.Group()


class Player(pygame.sprite.Sprite):
    Is_Rotated_Up = False
    Is_Rotated_Down = False
    Is_Rotated_Right = False
    Is_Rotated_Left = False

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        player_img = pygame.image.load(PLAYER_UP_IMG_PATH)
        self.image = player_img
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.speedx = 0
        self.speedy = 0

        self.rot = 0
        self.rot_speed = 5
        self.last_update = pygame.time.get_ticks()

    def update(self):
        self.speedx = 0
        self.speedy = 0

        key_state = pygame.key.get_pressed()

        if key_state[pygame.K_LEFT]:
            self.Is_Rotated_Up = False
            self.Is_Rotated_Down = False
            self.Is_Rotated_Right = False

            if not self.Is_Rotated_Left:
                self.image = pygame.image.load(PLAYER_LEFT_IMG_PATH)
                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Left = True

            self.speedx = -5
            self.speedy = 0

        if key_state[pygame.K_RIGHT]:
            self.Is_Rotated_Up = False
            self.Is_Rotated_Down = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Right:
                self.image = pygame.image.load(PLAYER_RIGHT_IMG_PATH)
                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Right = True

            self.speedx = 5
            self.speedy = 0

        if key_state[pygame.K_DOWN]:
            self.Is_Rotated_Up = False
            self.Is_Rotated_Right = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Down:
                self.image = pygame.image.load(PLAYER_DOWN_IMG_PATH)
                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Down = True

            self.speedy = 5
            self.speedx = 0

        if key_state[pygame.K_UP]:
            self.Is_Rotated_Down = False
            self.Is_Rotated_Right = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Up:
                self.image = pygame.image.load(PLAYER_UP_IMG_PATH)
                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Up = True

            self.speedy = -5
            self.speedx = 0

        self.rect.x += self.speedx
        self.rect.y += self.speedy

        hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(0.7))
        if hits:
            self.rect.x -= self.speedx
            self.rect.y -= self.speedy

    def shoot(self):
        direction = 'up'

        if self.Is_Rotated_Left:
            direction = 'left'
        if self.Is_Rotated_Right:
            direction = 'right'
        if self.Is_Rotated_Down:
            direction = 'down'
        if self.Is_Rotated_Up:
            direction = 'up'

        bullet = Bullet(self.rect.centerx, self.rect.centery, direction)
        SPRITES.add(bullet)
        PLAYER_BULLETS.add(bullet)


class Enemy(pygame.sprite.Sprite):
    Direction = 'down'
    Initial_rotate = True
    Last_shoot = 0

    def __init__(self, x, y, direction):
        pygame.sprite.Sprite.__init__(self)

        enemy_img = pygame.image.load(ENEMY_DOWN_IMG_PATH)
        self.image = enemy_img
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.Direction = direction

        self.speedx = 0
        self.speedy = 0

        self.rot = 0
        self.rot_speed = 5
        self.last_update = pygame.time.get_ticks()

    def update(self):
        self.speedx = 3
        self.speedy = 3

        frequency = random.randrange(1000, 10000)

        if self.Direction == 'down':
            self.image = pygame.image.load(ENEMY_DOWN_IMG_PATH)
            self.image.set_colorkey((255, 255, 255))

            self.rect.y += self.speedy

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(0.8))
            if hits:
                self.rect.y += -self.speedy

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'up'

        elif self.Direction == 'up':
            self.image = pygame.image.load(ENEMY_UP_IMG_PATH)
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += 0
            self.rect.y += -self.speedy

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(0.8))
            if hits:
                self.rect.y += self.speedy

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'down'

        elif self.Direction == 'left':
            self.image = pygame.image.load(ENEMY_LEFT_IMG_PATH)
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += -self.speedx

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(0.8))
            if hits:
                self.rect.x += self.speedx

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'down'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'up'

        elif self.Direction == 'right':
            self.image = pygame.image.load(ENEMY_RIGHT_IMG_PATH)
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += self.speedx

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(0.8))
            if hits:
                self.rect.x += -self.speedx

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'down'
                if d == 3:
                    self.Direction = 'up'

    def shoot(self):

        bullet = Bullet(self.rect.centerx, self.rect.centery, self.Direction)
        SPRITES.add(bullet)
        ENEMY_BULLETS.add(bullet)


class Brick(pygame.sprite.Sprite):

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(BRICK_IMG_PATH)

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)


class Bullet(pygame.sprite.Sprite):
    Direction = 'down'

    def __init__(self, x, y, direction):
        pygame.sprite.Sprite.__init__(self)

        bullet_img = pygame.image.load(BULLET_IMG_PATH)
        self.image = pygame.transform.scale(bullet_img, (10, 10))

        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.Direction = direction

        self.speedx = 0
        self.speedy = 0

    def update(self):
        self.speedx = 10
        self.speedy = 10

        if self.Direction == 'down':
            self.rect.y += self.speedy

        if self.Direction == 'up':
            self.rect.y += -self.speedy

        if self.Direction == 'left':
            self.rect.x += -self.speedx

        if self.Direction == 'right':
            self.rect.x += self.speedx


class Heart(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(HEART_IMG_PATH).convert()
        self.image = pygame.transform.scale(self.image, (30, 30))
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)


def fill_brick_coords():
    step = 32

    # outer borders
    x = 16
    y = 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 24):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 16
    y = 656
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 24):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 16
    y = 48
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 19):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 752
    y = 48
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 19):
        y += step
        BRICK_COORDINATES.append((x, y))

    # inner borders
    x = 112
    y = 48
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 5):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 112 + step
    y = 112
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 3):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 10
    y = 48
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = WIDTH - 16
    y = 48 + step * 5
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 9):
        x -= step
        BRICK_COORDINATES.append((x, y))

    x = WIDTH - 16 - step * 8
    y = 48 + step * 6
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = WIDTH - 16 - step * 9
    y = 48 + step * 6
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 2):
        x -= step
        BRICK_COORDINATES.append((x, y))

    x = 48
    y = 16 + step * 10
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 6):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48
    y = 16 + step * 14
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 6):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 4
    y = 16 + step * 13
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 1):
        y -= step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 5
    y = 16 + step * 17
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 3):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 8
    y = 16 + step * 9
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 9
    y = 16 + step * 12
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 3):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 8
    y = 16 + step * 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 11
    y = 16 + step * 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 12
    y = 16 + step * 17
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 2):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 16
    y = 16 + step * 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 18
    y = 16 + step * 15
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y -= step
        BRICK_COORDINATES.append((x, y))


def set_up():
    pygame.init()

    pygame.display.set_caption(NAME)
    icon = pygame.image.load(ICON_PATH)
    pygame.display.set_icon(icon)


def show_text(surface, text, size, color, x, y):
    font = pygame.font.Font(pygame.font.match_font('Raleway'), size)

    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)

    surface.blit(text_surface, text_rect)


def show_start(window, clock):
    background = pygame.image.load(START_IMG_PATH).convert()
    background_rect = background.get_rect()
    window.blit(background, background_rect)

    show_text(window, 'Fantastic game corporation presents', 32, (255, 255, 255), WIDTH / 2, HEIGHT / 4)
    show_text(window, 'TANKS', 64, (255, 255, 255), WIDTH / 2, HEIGHT / 2 - 25)
    show_text(window, 'Press a key to continue', 22, (255, 255, 255), WIDTH / 2, HEIGHT / 2 + 20)
    show_text(window, 'Created by Danylo Kotiai, IT-94', 30, (255, 255, 255), 160, HEIGHT - 50)

    pygame.display.flip()

    is_continue = False
    while not is_continue:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYUP:
                is_continue = True


def loop(window, clock):
    global IS_VICTORY
    global TIME
    global SPRITES
    global ENEMIES
    global BRICKS
    global PLAYER_BULLETS
    global ENEMY_BULLETS

    health = 3
    hearts = list()
    for h in HEART_COORDINATES:
        x, y = h
        heart = Heart(x, y)
        hearts.append(heart)
        SPRITES.add(heart)

    grass = pygame.image.load(GRASS_IMG_PATH).convert()
    grass = pygame.transform.scale(grass, (WIDTH, HEIGHT))
    grass_rect = grass.get_rect()

    player = Player(PLAYER_COORDINATES[0], PLAYER_COORDINATES[1])
    SPRITES.add(player)

    for e_coord in ENEMY_COORDINATES:
        x, y, direction = e_coord
        enemy = Enemy(x, y, direction)

        ENEMIES.add(enemy)
        SPRITES.add(enemy)

    for b_coord in BRICK_COORDINATES:
        x, y = b_coord
        brick = Brick(x, y)

        BRICKS.add(brick)
        SPRITES.add(brick)

    start_time = pygame.time.get_ticks()
    is_run = True
    while is_run:
        clock.tick(FPS)

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                is_run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    player.shoot()

        SPRITES.update()

        pygame.sprite.groupcollide(PLAYER_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(PLAYER_BULLETS, ENEMIES, True, True, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(ENEMY_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))

        if not ENEMIES:
            TIME = pygame.time.get_ticks() - start_time
            IS_VICTORY = True
            player.kill()
            break

        hits = pygame.sprite.spritecollide(player, ENEMY_BULLETS, True, pygame.sprite.collide_rect_ratio(0.8))
        if hits:
            health -= 1
            hearts[health].kill()
            if health == 0:
                TIME = pygame.time.get_ticks() - start_time
                IS_VICTORY = False
                SPRITES = pygame.sprite.Group()
                ENEMIES = pygame.sprite.Group()
                BRICKS = pygame.sprite.Group()
                PLAYER_BULLETS = pygame.sprite.Group()
                ENEMY_BULLETS = pygame.sprite.Group()
                break

        window.blit(grass, grass_rect)
        SPRITES.draw(window)

        pygame.display.flip()


def show_end(window, clock):
    img = pygame.image.load(DEFEAT_IMG_PATH)
    if IS_VICTORY:
        img = pygame.image.load(VICTORY_IMG_PATH)

    background = pygame.transform.scale(img, (WIDTH, HEIGHT)).convert()
    background_rect = background.get_rect()
    window.blit(background, background_rect)

    show_text(window, 'Do you want to restart? Press R', 32, (0, 0, 0), 200, 500)
    show_text(window, 'Your time: ' + str(TIME / 1000), 32, (0, 0, 0), 200, 550)

    pygame.display.flip()

    global IS_RESTART
    is_continue = False
    while not is_continue:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                IS_RESTART = False
                pygame.quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    IS_RESTART = True
                    is_continue = True


if __name__ == '__main__':
    fill_brick_coords()
    set_up()

    show_start(WINDOW, CLOCK)

    while IS_RESTART:
        loop(WINDOW, CLOCK)
        show_end(WINDOW, CLOCK)

    pygame.quit()
