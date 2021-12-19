import pygame
import random
from collections import deque
from queue import Queue, PriorityQueue
import math
import timeit
import csv
from sys import argv
from heapq import heappush, heappop
import matplotlib.pyplot as plt

import random, math
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

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

PLAYER_COORDINATES = (12, 16)
ENEMY_COORDINATES = [(9, 18, 'down'),
                     (8, 17, 'right'),
                     (15, 13, 'right')
                     # (4, 4, 'left'),
                     # (5, 5, 'down')
                     ]
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
INFINITY = 1000000000


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


def get_element_pos(x, y):
    grid_x, grid_y = x // CELL, y // CELL
    return grid_x, grid_y


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


GRAPH = 0


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


def astar(grath, start, stop):
    open_lst = {start}
    closed_lst = set([])
    poo = {}
    poo[start] = 0
    par = {}
    par[start] = start
    while len(open_lst) > 0:
        n = None
        for v in open_lst:
            if n == None or poo[v] + 1 < poo[n] + 1:
                n = v

        if n == None:
            print('Path does not exist!')
            return None

        if n == stop:
            reconst_path = []
            while par[n] != n:
                reconst_path.append(n)
                n = par[n]

            reconst_path.append(start)
            reconst_path.reverse()

            # print('Path found: {}'.format(reconst_path))
            return reconst_path

        temp = grath.get(n)
        for m in temp:
            if m not in open_lst and m not in closed_lst:
                open_lst.add(m)
                par[m] = n
                poo[m] = poo[n] + 1

            else:
                if poo[m] > poo[n] + 1:
                    poo[m] = poo[n] + 1
                    par[m] = n

                    if m in closed_lst:
                        closed_lst.remove(m)
                        open_lst.add(m)

        open_lst.remove(n)
        closed_lst.add(n)

    print('Path does not exist!')
    return None


def distance(node1: object, node2: object) -> object:
    y1 = ROWS - node1[1]
    y2 = ROWS - node2[1]
    return math.sqrt(math.pow((node1[0] - node2[0]), 2) + math.pow((y1 - y2), 2))


def get_closest_enemy(node):
    shortest_distance = INFINITY
    coords = (0, 0)

    for enemy in ENEMIES:
        enemy_node = get_element_pos(enemy.rect.x, enemy.rect.y)
        dist = distance(node, enemy_node)
        if dist < shortest_distance:
            shortest_distance = dist
            coords = (enemy.rect.x, enemy.rect.y)

    return coords, shortest_distance


def distance_to_closest_enemy(node, goal):
    if node == goal:
        return -INFINITY

    dist = (get_closest_enemy(node))[1]
    return dist


def evaluation_function(node, goal):
    return distance_to_closest_enemy(node, goal) * -1


def minimax(maximizing, depth, node, alpha, beta, goal):
    if depth == 3:
        return evaluation_function(node, goal)
    if maximizing:
        value = -INFINITY
        for child in GRAPH[node]:
            value = max(value, minimax(False, depth, child, alpha, beta, goal))
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return value
    else:
        value = INFINITY
        for child in GRAPH[node]:
            value = min(value, minimax(True, depth + 1, child, alpha, beta, goal))
            alpha = min(alpha, value)
            if beta <= alpha:
                break
        return value


def expectimax(maximizing, depth, node, goal):
    if depth == 3:
        return evaluation_function(node, goal)
    if maximizing:
        return max(expectimax(False, depth, child, goal) for child in GRAPH[node])
    else:
        return sum(expectimax(True, depth + 1, child, goal) for child in GRAPH[node]) / float(len(GRAPH[node]))

SCORE = 0

ACTIONS = 5
STATECOUNT = 12


def CaptureNormalisedState(PlayerXPos, PlayerYPos, enms):
    gstate = np.zeros([STATECOUNT])
    gstate[0] = PlayerXPos
    gstate[1] = PlayerYPos

    i = 2
    for e in enms:
        gstate[i] = e[0]
        i += 1
        gstate[i] = e[1]
        i += 1

    return gstate


class Brain:
    def __init__(self, NbrStates, NbrActions):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions

        self.model = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=64, activation='relu', input_dim=self.NbrStates))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=self.NbrActions,
                        activation='linear'))

        model.compile(loss='mse', optimizer='adam')

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epoch, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.predict(s.reshape(1, self.NbrStates)).flatten()


class ExpReplay:
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)



ExpReplay_CAPACITY = 2000
OBSERVEPERIOD = 750
BATCH_SIZE = 128
GAMMA = 0.95
MAX_EPSILON = 1
MIN_EPSILON = 0.05
LAMBDA = 0.0005


class Agent:
    def __init__(self, NbrStates, NbrActions):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions

        self.brain = Brain(NbrStates, NbrActions)
        self.ExpReplay = ExpReplay(ExpReplay_CAPACITY)
        self.steps = 0
        self.epsilon = MAX_EPSILON


    def Act(self, s):
        if (random.random() < self.epsilon or self.steps < OBSERVEPERIOD):
            return random.randint(0, self.NbrActions - 1)
        else:
            return np.argmax(self.brain.predictOne(s))

    def CaptureSample(self, sample):
        self.ExpReplay.add(sample)

        self.steps += 1
        if (self.steps > OBSERVEPERIOD):
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * (self.steps - OBSERVEPERIOD))

    def Process(self):
        batch = self.ExpReplay.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = np.zeros(self.NbrStates)

        states = np.array([batchitem[0] for batchitem in batch])
        states_ = np.array([(no_state if batchitem[3] is None else batchitem[3]) for batchitem in batch])

        predictedQ = self.brain.predict(states)
        predictedNextQ = self.brain.predict(states_)

        x = np.zeros((batchLen, self.NbrStates))
        y = np.zeros((batchLen, self.NbrActions))

        for i in range(batchLen):
            batchitem = batch[i]
            state = batchitem[0];
            a = batchitem[1];
            reward = batchitem[2];
            nextstate = batchitem[3]

            targetQ = predictedQ[i]
            if nextstate is None:
                targetQ[a] = reward
            else:
                targetQ[a] = reward + GAMMA * np.amax(predictedNextQ[i])

            x[i] = state
            y[i] = targetQ

        self.brain.train(x, y)


class Player(pygame.sprite.Sprite):
    Is_Rotated_Up = False
    Is_Rotated_Down = False
    Is_Rotated_Right = False
    Is_Rotated_Left = False
    Last_shoot = 0
    Frequency = 300
    Direction = 'down'
    Goal = (0, 0)
    BestAction = 2

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

        self.Goal = (self.rect.x // CELL, self.rect.y // CELL)

    def update(self):
        # global SCORE

        self.speedx = 3
        self.speedy = 3

        # graph = get_graph()
        # self.astar_path(graph)
        # self.move_by_minimax(self.rect.x, self.rect.y)

        # start = (self.rect.x // CELL, self.rect.y // CELL)
        # if start == self.Goal:
        #     SCORE += 1
        #     self.set_goal()

        if self.BestAction == 4:

            for e in ENEMIES:
                if self.rect.x - 30 <= e.rect.x <= self.rect.x + 30 and e.rect.y <= self.rect.y:
                    self.shoot_with_delay()

        if self.BestAction == 0:
            self.Direction = 'up'
            self.image = pygame.transform.scale(pygame.image.load(PLAYER_UP_IMG_PATH), (30, 30))
            self.rect.y += -self.speedy

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                self.rect.y += self.speedy

            for e in ENEMIES:
                if self.rect.x - 30 <= e.rect.x <= self.rect.x + 30 and e.rect.y <= self.rect.y:
                    self.shoot_with_delay()

        if self.BestAction == 1:
            self.Direction = 'right'
            self.image = pygame.transform.scale(pygame.image.load(PLAYER_RIGHT_IMG_PATH), (30, 30))
            self.rect.x += self.speedx

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                self.rect.x += -self.speedx

            for e in ENEMIES:
                if self.rect.y - 30 <= e.rect.y <= self.rect.y + 30 and e.rect.x >= self.rect.x:
                    self.shoot_with_delay()

        if self.BestAction == 2:
            self.Direction = 'down'
            self.image = pygame.transform.scale(pygame.image.load(PLAYER_DOWN_IMG_PATH), (30, 30))
            self.rect.y += self.speedy

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                self.rect.y += -self.speedy

            for e in ENEMIES:
                if self.rect.x - 30 <= e.rect.x <= self.rect.x + 30 and e.rect.y >= self.rect.y:
                    self.shoot_with_delay()

        if self.BestAction == 3:
            self.Direction = 'left'
            self.image = pygame.transform.scale(pygame.image.load(PLAYER_LEFT_IMG_PATH), (30, 30))
            self.rect.x += -self.speedx

            hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
            if hits:
                self.rect.x += self.speedx

            for e in ENEMIES:
                if self.rect.y - 30 <= e.rect.y <= self.rect.y + 30 and e.rect.x <= self.rect.x:
                    self.shoot_with_delay()

        # pygame.draw.rect(WINDOW, pygame.Color('black'), get_rect(*self.Goal), CELL, border_radius=CELL // 10)
        self.image.set_colorkey((255, 255, 255))


    def move_by_minimax(self, x, y):
        # alg = 'expectimax'
        # alpha = -INFINITY
        # beta = INFINITY
        # node = get_element_pos(x, y)
        # scores = []
        #
        # if alg == 'minimax':
        #     scores = [minimax(True, 0, child, alpha, beta, self.Goal) for child in GRAPH[node]]
        # else:
        #     scores = [expectimax(True, 0, child, self.Goal) for child in GRAPH[node]]
        #
        # index = scores.index(max(scores))
        # optimal = (GRAPH[node])[index]
        optimal = 0
        self.path(GRAPH, optimal)

    def set_goal(self):
        c = 0
        r = 0
        is_right_value = False
        while not is_right_value:
            c = random.randrange(3, 20)
            r = random.randrange(3, 20)
            if GRID[r][c] != 1:
                is_right_value = True

        self.Goal = (c, r)

    def path(self, graph, node):
        global SCORE

        start = (self.rect.x // CELL, self.rect.y // CELL)
        if start == self.Goal:
            SCORE += 1
            self.set_goal()

        path = astar(graph, start, self.Goal)

        # for cell in path:
        #     pygame.draw.rect(WINDOW, pygame.Color('red'), get_rect(*cell), CELL, border_radius=CELL // 3)

        first = path[0]
        second = path[1]
        last = path[-1]
        pygame.draw.rect(WINDOW, pygame.Color('black'), get_rect(*last), CELL, border_radius=CELL // 10)

        x_s, y_s = start
        x_g, y_g = second
        x_f, y_f = first

        x_check, y_check = (x_f * CELL, y_f * CELL)
        checking_x = x_check - self.rect.x
        checking_y = y_check - self.rect.y
        first = node
        if -3 <= checking_x <= 3 and -3 <= checking_y <= 3:
            if x_s < x_g:
                self.Direction = 'right'
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_RIGHT_IMG_PATH), (30, 30))
                self.rect.x += self.speedx

                for e in ENEMIES:
                    if self.rect.y - 3 <= e.rect.y <= self.rect.y + 3 and e.rect.x >= self.rect.x:
                        self.shoot_with_delay()
            elif x_s > x_g:
                self.Direction = 'left'
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_LEFT_IMG_PATH), (30, 30))
                self.rect.x += -self.speedx

                for e in ENEMIES:
                    if self.rect.y - 3 <= e.rect.y <= self.rect.y + 3 and e.rect.x <= self.rect.x:
                        self.shoot_with_delay()
            elif y_s < y_g:
                self.Direction = 'down'
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_DOWN_IMG_PATH), (30, 30))
                self.rect.y += self.speedy

                for e in ENEMIES:
                    if self.rect.x - 3 <= e.rect.x <= self.rect.x + 3 and e.rect.y >= self.rect.y:
                        self.shoot_with_delay()
            elif y_s > y_g:
                self.Direction = 'up'
                self.image = pygame.transform.scale(pygame.image.load(PLAYER_UP_IMG_PATH), (30, 30))
                self.rect.y += -self.speedy

                for e in ENEMIES:
                    if self.rect.x - 3 <= e.rect.x <= self.rect.x + 3 and e.rect.y <= self.rect.y:
                        self.shoot_with_delay()
        else:
            if self.Direction == 'right':
                self.rect.x += self.speedx
            elif self.Direction == 'left':
                self.rect.x += -self.speedx
            elif self.Direction == 'down':
                self.rect.y += self.speedy
            elif self.Direction == 'up':
                self.rect.y += -self.speedy

    def shoot(self):
        # direction = 'up'
        #
        # if self.Is_Rotated_Left:
        #     direction = 'left'
        # if self.Is_Rotated_Right:
        #     direction = 'right'
        # if self.Is_Rotated_Down:
        #     direction = 'down'
        # if self.Is_Rotated_Up:
        #     direction = 'up'

        bullet = Bullet(self.rect.centerx, self.rect.centery, self.Direction)
        SPRITES.add(bullet)
        PLAYER_BULLETS.add(bullet)

    def shoot_with_delay(self):
        now = pygame.time.get_ticks()
        if now - self.Last_shoot > self.Frequency:
            self.shoot()
            self.Last_shoot = now


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
            self.image = pygame.image.load(ENEMY_DOWN_IMG_PATH)
            self.image.set_colorkey((255, 255, 255))

            self.rect.y += self.speedy

            now = pygame.time.get_ticks()
            if now - self.Last_shoot > self.Frequency:
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
            if now - self.Last_shoot > self.Frequency:
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
            if now - self.Last_shoot > self.Frequency:
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
            if now - self.Last_shoot > self.Frequency:
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

        # now = pygame.time.get_ticks()
        # if now - self.Last_shoot > self.Frequency:
        #     self.shoot()
        #     self.Last_shoot = now
        #
        # graph = get_graph()
        # if self.Alg == 1:
        #     self.bfs_path(graph)
        # elif self.Alg == 2:
        #     self.dfs_path(graph)
        # elif self.Alg == 3:
        #     self.uniform_cost_search_path(graph)
        #
        #
        # self.image.set_colorkey((255, 255, 255))

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

        i = 1
        cell = goal
        p_cell = start
        while cell and cell in visited:
            i += 1
            if i == count - 1:
                previous = cell
            p_cell = cell
            cell = visited[cell]

        # pygame.draw.rect(WINDOW, pygame.Color('black'), get_rect(*previous), CELL, border_radius=CELL // 10)

        xE, yE = start
        xC, yC = previous
        xP, yP = p_cell

        if xE < xC:
            self.Direction = 'right'
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_RIGHT_IMG_PATH), (30, 30))
            self.rect.x += self.speedx
        elif xE > xC:
            self.Direction = 'left'
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_LEFT_IMG_PATH), (30, 30))
            self.rect.x += -self.speedx
        elif yE < yC:
            self.Direction = 'down'
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_DOWN_IMG_PATH), (30, 30))
            self.rect.y += self.speedy
        elif yE > yC:
            self.Direction = 'up'
            self.image = pygame.transform.scale(pygame.image.load(ENEMY_UP_IMG_PATH), (30, 30))
            self.rect.y += -self.speedy

        hits = pygame.sprite.spritecollide(self, BRICKS, False, pygame.sprite.collide_rect_ratio(1))
        if hits:
            self.rect.x = xP * CELL
            self.rect.y = yP * CELL

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
    for i in range(1, 15):
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
    for i in range(1, 4):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 8
    y = 16 + step * 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 4):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 11
    y = 16 + step * 11
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 9):
        y += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 12
    y = 16 + step * 17
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 2):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 12
    y = 16 + step * 16
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 10):
        x += step
        BRICK_COORDINATES.append((x, y))

    x = 48 + step * 18
    y = 16 + step * 15
    BRICK_COORDINATES.append((x, y))
    for i in range(1, 9):
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


def save_stat(result, score, time, alg):
    file = open('stat.csv', 'a', newline='')
    writer = csv.writer(file)
    writer.writerow([int(result), str(time / 1000), str(score), alg])
    file.close()

HISTORY = []

def loop(window, clock, the_agent, game_state, game_history):
    global IS_VICTORY
    global TIME
    global SPRITES
    global ENEMIES
    global BRICKS
    global PLAYER_BULLETS
    global ENEMY_BULLETS
    global GRID
    global GRAPH
    global SCORE

    grass = pygame.image.load(GRASS_IMG_PATH).convert()
    grass = pygame.transform.scale(grass, (WIDTH, HEIGHT))
    grass_rect = grass.get_rect()

    bricks = list()
    for b_coord in BRICK_COORDINATES:
        brick = Brick(b_coord[0], b_coord[1])
        BRICKS.add(brick)
        SPRITES.add(brick)

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

    iteration = 0
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
                    alg += 1
                    if alg == 4:
                        alg = 1

        BestAction = the_agent.Act(game_state)
        player.BestAction = BestAction

        window.blit(grass, grass_rect)
        SPRITES.update()

        pygame.sprite.groupcollide(PLAYER_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(PLAYER_BULLETS, ENEMIES, True, True, pygame.sprite.collide_rect_ratio(0.8))
        pygame.sprite.groupcollide(ENEMY_BULLETS, BRICKS, True, False, pygame.sprite.collide_rect_ratio(0.8))

        # enms.clear()
        # for e in ENEMIES:
        #     enms.append((e.rect.x, e.rect.y))

        if not ENEMIES:
            TIME = pygame.time.get_ticks() - start_time
            IS_VICTORY = True

            save_stat(IS_VICTORY, 5 - len(ENEMIES), TIME, 1)

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

                save_stat(IS_VICTORY, 5 - len(ENEMIES), TIME, 'expectimax')

                player.kill()

                SPRITES = pygame.sprite.Group()
                ENEMIES = pygame.sprite.Group()
                BRICKS = pygame.sprite.Group()
                PLAYER_BULLETS = pygame.sprite.Group()
                ENEMY_BULLETS = pygame.sprite.Group()
                break

        SPRITES.draw(window)

        pygame.display.flip()

        NextState = CaptureNormalisedState(player.rect.x, player.rect.y, enms)

        the_agent.CaptureSample((game_state, BestAction, 3 - len(ENEMIES), NextState))
        the_agent.Process()
        game_state = NextState
        iteration += 1

        if iteration % 15 == 0:
            game_history.append((iteration, 3 - len(ENEMIES)))

    return game_history



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

    # global IS_RESTART
    # is_continue = False
    # while not is_continue:
    #     clock.tick(FPS)
    #
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             IS_RESTART = False
    #             pygame.quit()
    #             break
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_r:
    #                 IS_RESTART = True
    #                 is_continue = True


if __name__ == '__main__':
    fill_brick_coords()
    set_up()

    show_start(WINDOW, CLOCK)

    # while IS_RESTART:
    while True:
        player = Player(PLAYER_COORDINATES[0], PLAYER_COORDINATES[1])
        SPRITES.add(player)

        for e_coord in ENEMY_COORDINATES:
            x, y, direction = e_coord
            enemy = Enemy(x, y, direction, player)

            ENEMIES.add(enemy)
            SPRITES.add(enemy)

        enms = []
        for e in ENEMIES:
            enms.append((e.rect.x, e.rect.y))

        STATECOUNT = len(enms) * 2 + 2

        GameHistory = []
        TheAgent = Agent(STATECOUNT, ACTIONS)
        GameState = CaptureNormalisedState(player.rect.x, player.rect.y, enms)

        history = loop(WINDOW, CLOCK, TheAgent, GameState, GameHistory)
        show_end(WINDOW, CLOCK)

        x_val = [x[0] for x in history]
        y_val = [x[1] for x in history]

        print(history)
        print(x_val)
        print(y_val)
        plt.plot(x_val, y_val)
        plt.xlabel("Time")
        plt.ylabel("Score")
        plt.show()

    pygame.quit()
