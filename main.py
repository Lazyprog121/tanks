import pygame
import random
from collections import deque
from queue import Queue, PriorityQueue
from sys import argv
from heapq import heappush, heappop

WIDTH = 768
HEIGHT = 672
FPS = 30

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
CLOCK = pygame.time.Clock()

IS_RESTART = True
IS_VICTORY = True
TIME = 0

COLUMNS = 24
ROWS = 21
CELL = 32

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

PLAYER_COORDINATES = (12, 19)
ENEMY_COORDINATES = [(1, 2, 'down'),
                     (2, 2, 'up'),
                     (3, 3, 'right'),
                     (4, 4, 'left'),
                     (5, 5, 'down')]
BRICK_COORDINATES = list()
HEART_COORDINATES = [(WIDTH - 60, 50),
                     (WIDTH - 100, 50),
                     (WIDTH - 140, 50)]

SPRITES = pygame.sprite.Group()
ENEMIES = pygame.sprite.Group()
BRICKS = pygame.sprite.Group()
PLAYER_BULLETS = pygame.sprite.Group()
ENEMY_BULLETS = pygame.sprite.Group()


GRID = [[0] * COLUMNS for i in range(ROWS)]


def fill_grid():
    for r in range(ROWS):
        for c in range(COLUMNS):
            if random.random() < 0.2:
                GRID[r][c] = 1

    for c in range(COLUMNS):
        GRID[0][c] = 1
        GRID[ROWS - 1][c] = 1

    for r in range(ROWS):
        GRID[r][0] = 1
        GRID[r][COLUMNS - 1] = 1

    for e_coords in ENEMY_COORDINATES:
        x, y, d = e_coords
        GRID[x][y] = 0

    x, y = PLAYER_COORDINATES
    GRID[x][y] = 0


def get_rect(x, y):
    return x * CELL + 1, y * CELL + 1, CELL - 2, CELL - 2


def get_next_nodes(x, y):
    checking_next_node = lambda x, y: True if 0 <= x < COLUMNS and 0 <= y < ROWS and not GRID[y][x] else False
    paths = [-1, 0], [0, -1], [1, 0], [0, 1]
    return [(x + dx, y + dy) for dx, dy in paths if checking_next_node(x + dx, y + dy)]


def get_graph():
    graph = {}
    for y, row in enumerate(GRID):
        for x, col in enumerate(row):
            if not col:
                graph[(x, y)] = graph.get((x, y), []) + get_next_nodes(x, y)
    return graph


def bfs(enemy, user, graph):
    queue = deque([enemy])
    visited = {enemy: None}

    while queue:
        current = queue.popleft()
        if current == user:
            break

        next_nodes = graph[current]
        for next in next_nodes:
            if next not in visited:
                queue.append(next)
                visited[next] = current
    return queue, visited


def dfs(enemy, user, graph):
    stack = [(enemy, [enemy])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex not in visited:
            if vertex == user:
                return path
            visited.add(vertex)
            for neighbor in graph[vertex]:
                stack.append((neighbor, path + [neighbor]))


def ucs(graph, start, end):
    queue = PriorityQueue()
    queue.put((0, start))
    visited = {start: None}

    while True:
        ucs_w, current_node = queue.get()
        if current_node == end:
            break

        for node in graph[current_node]:
            if node not in visited:
                queue.put((ucs_w, node))
                visited[node] = current_node

    return visited


class Player(pygame.sprite.Sprite):
    Is_Rotated_Up = False
    Is_Rotated_Down = False
    Is_Rotated_Right = False
    Is_Rotated_Left = False

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.transform.scale(pygame.image.load(PLAYER_UP_IMG_PATH), (30, 30))
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.x = y * CELL
        self.rect.y = x * CELL
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
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_LEFT_IMG_PATH), (30, 30))

                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Left = True

            self.speedx = -5
            self.speedy = 0

        if key_state[pygame.K_RIGHT]:
            self.Is_Rotated_Up = False
            self.Is_Rotated_Down = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Right:
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_RIGHT_IMG_PATH), (30, 30))

                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Right = True

            self.speedx = 5
            self.speedy = 0

        if key_state[pygame.K_DOWN]:
            self.Is_Rotated_Up = False
            self.Is_Rotated_Right = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Down:
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_DOWN_IMG_PATH), (30, 30))

                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Down = True

            self.speedy = 5
            self.speedx = 0

        if key_state[pygame.K_UP]:
            self.Is_Rotated_Down = False
            self.Is_Rotated_Right = False
            self.Is_Rotated_Left = False

            if not self.Is_Rotated_Up:
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_UP_IMG_PATH), (30, 30))

                self.image.set_colorkey((255, 255, 255))

                self.Is_Rotated_Up = True

            self.speedy = -5
            self.speedx = 0

        self.rect.x += self.speedx
        self.rect.y += self.speedy

        hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        if hits:
            if self.Is_Rotated_Up:
                y = hits[0].rect.bottom
                self.rect.top = y + 1
            if self.Is_Rotated_Down:
                y = hits[0].rect.top
                self.rect.bottom = y - 1
            if self.Is_Rotated_Right:
                x = hits[0].rect.left
                self.rect.right = x - 1
            if self.Is_Rotated_Left:
                x = hits[0].rect.right
                self.rect.left = x + 1
            # self.rect.x -= self.speedx
            # self.rect.y -= self.speedy

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
    Frequency = 0
    Alg = 1

    Player = 0

    def __init__(self, x, y, direction, player):
        pygame.sprite.Sprite.__init__(self)

        enemy_img = pygame.image.load(ENEMY_DOWN_IMG_PATH)
        self.image = pygame.transform.scale(enemy_img, (30, 30))
        self.image.set_colorkey((255, 255, 255))

        self.rect = self.image.get_rect()
        self.rect.x = y * CELL
        self.rect.y = x * CELL
        self.Direction = direction

        self.speedx = 0
        self.speedy = 0

        self.rot = 0
        self.rot_speed = 5
        self.last_update = pygame.time.get_ticks()

        self.Player = player
        self.Frequency = random.randrange(1000, 10000)

    def update(self):
        self.speedx = 3
        self.speedy = 3

        if self.Direction == 'down':
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_DOWN_IMG_PATH), (30, 30))
            self.image.set_colorkey((255, 255, 255))

            self.rect.y += self.speedy

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > self.Frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                y = hits[0].rect.top
                self.rect.bottom = y - 1

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'up'

        elif self.Direction == 'up':
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_UP_IMG_PATH), (30, 30))
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += 0
            self.rect.y += -self.speedy

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > self.Frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                y = hits[0].rect.bottom
                self.rect.top = y + 1

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'down'

        elif self.Direction == 'left':
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_LEFT_IMG_PATH), (30, 30))
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += -self.speedx

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > self.Frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                x = hits[0].rect.right
                self.rect.left = x + 1

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'down'
                if d == 2:
                    self.Direction = 'right'
                if d == 3:
                    self.Direction = 'up'

        elif self.Direction == 'right':
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_RIGHT_IMG_PATH), (30, 30))
            self.image.set_colorkey((255, 255, 255))

            self.rect.x += self.speedx

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > self.Frequency:
                self.shoot()
                self.Last_shoot = now

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                x = hits[0].rect.left
                self.rect.right = x - 1

                d = random.randrange(1, 4)
                if d == 1:
                    self.Direction = 'left'
                if d == 2:
                    self.Direction = 'down'
                if d == 3:
                    self.Direction = 'up'

        graph = get_graph()
        if self.Alg == 1:
            self.bfs_path(graph)
        elif self.Alg == 2:
            self.dfs_path(graph)
        elif self.Alg == 3:
            self.uniform_cost_search_path(graph)

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.centery, self.Direction)
        SPRITES.add(bullet)
        ENEMY_BULLETS.add(bullet)

    def set_alg(self, alg):
        self.Alg = alg

    def bfs_path(self, graph):
        start = (self.rect.x // CELL, self.rect.y // CELL)
        user = (self.Player.rect.x // CELL, self.Player.rect.y // CELL)
        queue, visited = bfs(start, user, graph)
        goal = user

        count = 1
        cell = goal
        previous = goal
        while cell and cell in visited:
            count += 1
            pygame.draw.rect(WINDOW, pygame.Color('yellow'), get_rect(*cell), CELL, border_radius=CELL // 2)
            cell = visited[cell]

        # i = 1
        # cell = goal
        # while cell and cell in visited:
        #     i += 1
        #     if i == count - 1:
        #         previous = cell
        #     p_cell = cell
        #     cell = visited[cell]

        # pygame.draw.rect(WINDOW, pygame.Color('black'), get_rect(*previous), CELL, border_radius=CELL // 10)

        # xE, yE = start
        # xC, yC = previous
        # if xE < xC:
        #     self.Direction = 'right'
        #     # hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        #     # if hits:
        #     #     x = hits[0].rect.left
        #     #     self.rect.right = x - 1
        #     #
        #     #     d = random.randrange(1, 4)
        #     #     if d == 1:
        #     #         self.Direction = 'left'
        #     #     if d == 2:
        #     #         self.Direction = 'down'
        #     #     if d == 3:
        #     #         self.Direction = 'up'
        # elif xE > xC:
        #     self.Direction = 'left'
        #     # hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        #     # if hits:
        #     #     x = hits[0].rect.right
        #     #     self.rect.left = x + 1
        #     #
        #     #     d = random.randrange(1, 4)
        #     #     if d == 1:
        #     #         self.Direction = 'down'
        #     #     if d == 2:
        #     #         self.Direction = 'right'
        #     #     if d == 3:
        #     #         self.Direction = 'up'
        # elif yE < yC:
        #     self.Direction = 'down'
        #     # hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        #     # if hits:
        #     #     y = hits[0].rect.top
        #     #     self.rect.bottom = y - 1
        #     #
        #     #     d = random.randrange(1, 4)
        #     #     if d == 1:
        #     #         self.Direction = 'left'
        #     #     if d == 2:
        #     #         self.Direction = 'right'
        #     #     if d == 3:
        #     #         self.Direction = 'up'
        # elif yE > yC:
        #     self.Direction = 'up'
        #     # hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        #     # if hits:
        #     #     y = hits[0].rect.bottom
        #     #     self.rect.top = y + 1
        #     #
        #     #     d = random.randrange(1, 4)
        #     #     if d == 1:
        #     #         self.Direction = 'left'
        #     #     if d == 2:
        #     #         self.Direction = 'right'
        #     #     if d == 3:
        #     #         self.Direction = 'down'

    def dfs_path(self, graph):
        start = (self.rect.x // CELL, self.rect.y // CELL)

        user = (self.Player.rect.x // CELL, self.Player.rect.y // CELL)
        path = dfs(start, user, graph)

        for cell in path:
            pygame.draw.rect(WINDOW, pygame.Color('yellow'), get_rect(*cell), CELL, border_radius=CELL // 2)

    def uniform_cost_search_path(self, graph):
        start = (self.rect.x // CELL, self.rect.y // CELL)
        user = (self.Player.rect.x // CELL, self.Player.rect.y // CELL)
        cost = graph
        answer = ucs(graph, start, user)
        goal = user

        count = 1
        cell = goal
        while cell and cell in answer:
            count += 1
            pygame.draw.rect(WINDOW, pygame.Color('yellow'), get_rect(*cell), CELL, border_radius=CELL // 2)
            cell = answer[cell]


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
    # x = 112
    # y = 48
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 5):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 112 + step
    # y = 112
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 3):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 10
    # y = 48
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = WIDTH - 16
    # y = 48 + step * 5
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 9):
    #     x -= step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = WIDTH - 16 - step * 8
    # y = 48 + step * 6
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = WIDTH - 16 - step * 9
    # y = 48 + step * 6
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 2):
    #     x -= step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48
    # y = 16 + step * 10
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 6):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48
    # y = 16 + step * 14
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 6):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 4
    # y = 16 + step * 13
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 1):
    #     y -= step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 5
    # y = 16 + step * 17
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 3):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 8
    # y = 16 + step * 9
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 9
    # y = 16 + step * 12
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 3):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 8
    # y = 16 + step * 16
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 11
    # y = 16 + step * 16
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 12
    # y = 16 + step * 17
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 2):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 16
    # y = 16 + step * 16
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     x += step
    #     BRICK_COORDINATES.append((x, y))
    #
    # x = 48 + step * 18
    # y = 16 + step * 15
    # BRICK_COORDINATES.append((x, y))
    # for i in range(1, 4):
    #     y -= step
    #     BRICK_COORDINATES.append((x, y))


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
    global GRID

    grass = pygame.image.load(GRASS_IMG_PATH).convert()
    grass = pygame.transform.scale(grass, (WIDTH, HEIGHT))
    grass_rect = grass.get_rect()

    player = Player(PLAYER_COORDINATES[0], PLAYER_COORDINATES[1])
    SPRITES.add(player)

    for e_coord in ENEMY_COORDINATES:
        x, y, direction = e_coord
        enemy = Enemy(x, y, direction, player)

        ENEMIES.add(enemy)
        SPRITES.add(enemy)
        # break

    GRID = [[0] * COLUMNS for i in range(ROWS)]
    fill_grid()
    r_counter = 0.5
    for r in GRID:
        c_counter = 0.5

        for c in r:
            if c == 1:
                brick = Brick(c_counter * CELL, r_counter * CELL)
                BRICKS.add(brick)
                SPRITES.add(brick)
            c_counter += 1

        r_counter += 1

    health = 3
    hearts = list()
    for h in HEART_COORDINATES:
        x, y = h
        heart = Heart(x, y)
        hearts.append(heart)
        SPRITES.add(heart)

    alg = 2
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
                if event.key == pygame.K_z:
                    for e in ENEMIES:
                        e.set_alg(alg)
                        print(alg)
                    alg += 1
                    if alg == 4:
                        alg = 1

        window.blit(grass, grass_rect)
        SPRITES.update()

        pygame.sprite.groupcollide(PLAYER_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(PLAYER_BULLETS, ENEMIES, True, True, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(ENEMY_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))

        if not ENEMIES:
            TIME = pygame.time.get_ticks() - start_time
            IS_VICTORY = True
            player.kill()
            SPRITES = pygame.sprite.Group()
            ENEMIES = pygame.sprite.Group()
            BRICKS = pygame.sprite.Group()
            PLAYER_BULLETS = pygame.sprite.Group()
            ENEMY_BULLETS = pygame.sprite.Group()
            break

        hits = pygame.sprite.spritecollide(player, ENEMY_BULLETS, True, pygame.sprite.collide_rect_ratio(0.8))
        if hits:
            health -= 1
            hearts[health].kill()
            if health == 0:
                TIME = pygame.time.get_ticks() - start_time
                IS_VICTORY = False
                player.kill()
                SPRITES = pygame.sprite.Group()
                ENEMIES = pygame.sprite.Group()
                BRICKS = pygame.sprite.Group()
                PLAYER_BULLETS = pygame.sprite.Group()
                ENEMY_BULLETS = pygame.sprite.Group()
                break

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
