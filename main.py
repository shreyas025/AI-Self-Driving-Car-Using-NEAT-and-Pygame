import pygame
import os
import math
import sys
import neat

# Constants
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
COLLISION_COLOR = pygame.Color(2, 105, 31, 255)
WHITE = (255, 255, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
TRACK = pygame.image.load(os.path.join("Assets", "track.png"))


class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car1.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.check_collision()
        return self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def check_collision(self):
        length = 40
        collision_points = [
            [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
             int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)],
            [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
             int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]
        ]

        for point in collision_points:
            if SCREEN.get_at(point) == COLLISION_COLOR:
                self.alive = False

            # Draw Collision Points
            pygame.draw.circle(SCREEN, CYAN, point, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        elif self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x, y = self.rect.center

        while not SCREEN.get_at((x, y)) == COLLISION_COLOR and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
        pygame.draw.line(SCREEN, WHITE, self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, GREEN, (x, y), 3)

        dist = int(math.sqrt((self.rect.center[0] - x) ** 2 + (self.rect.center[1] - y) ** 2))
        self.radars.append([radar_angle, dist])

    def data(self):
        input = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input[i] = int(radar[1])
        return input


def remove(index):
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config):
    global cars, ge, nets

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    main_game_loop()


def main_game_loop():
    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            ge[i].fitness += 1
            if not car.sprite.alive:
                remove(i)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.sprite.update())
            car.sprite.direction = 1 if output[0] > 0.7 else -1 if output[1] > 0.7 else 0

        for car in cars:
            car.draw(SCREEN)
            car.update()

        pygame.display.update()


def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)