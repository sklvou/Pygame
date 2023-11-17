import pygame
import random
from settings import TILE_SIZE

def load_and_scale_img(path, size):
    return pygame.transform.scale(pygame.image.load(path), (size, size))


class Map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = self.create_grid(width, height)
        # 画像リソースのロード
        self.grass_img = load_and_scale_img('img/map/grass.png', TILE_SIZE)
        self.river_img = load_and_scale_img('img/map/river.png', TILE_SIZE)
        self.bridge_img = load_and_scale_img('img/map/bridge.png', TILE_SIZE)

    # マップの生成
    def create_grid(self, width, height):
        grid = [[0 for _ in range(width)] for _ in range(height)]  # 最初は全て草原で初期化

        # 川を曲がりくねらせながら描画
        river_path = []
        x = width // 4
        for y in range(height):
            river_path.append((x, y))
            grid[y][x] = 1  # 川をセット
            if random.choice([True, False]):  # ランダムに川の流れを左右に変える
                x += random.choice([-1, 1])
                x = max(0, min(x, width - 1))  # グリッドの範囲内に収める

        # 橋をランダムな位置に配置
        for _ in range(3):  # 3つの橋を配置
            bridge_pos = random.choice(river_path)
            grid[bridge_pos[1]][bridge_pos[0]] = 2

        return grid

    def draw(self, screen):
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile == 0:  # 草原
                    screen.blit(self.grass_img, (x * TILE_SIZE, y * TILE_SIZE))
                elif tile == 1:  # 川
                    screen.blit(self.river_img, (x * TILE_SIZE, y * TILE_SIZE))
                elif tile == 2:  # 橋
                    screen.blit(self.bridge_img, (x * TILE_SIZE, y * TILE_SIZE))
