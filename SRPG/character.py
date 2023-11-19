import random
import pygame
from settings import TILE_SIZE
from map import load_and_scale_img

class Character:
    def __init__(self, x, y, img_path):
        self.x = self.initial_x = x
        self.y = self.initial_y = y
        self.image = load_and_scale_img(img_path, TILE_SIZE)
        self.rect = self.image.get_rect(topleft=(self.x * TILE_SIZE, self.y * TILE_SIZE))
        self.has_moved = False

    def draw(self, screen):
        screen.blit(self.image, (self.x * TILE_SIZE, self.y * TILE_SIZE))

    def move(self, dx, dy):
        if not self.has_moved:
            self.x += dx
            self.y += dy
            self.rect.x = self.x * TILE_SIZE
            self.rect.y = self.y * TILE_SIZE
            self.has_moved = True
    
    def reset_movement(self):
        self.x = self.initial_x
        self.y = self.initial_y
        self.rect.x = self.x * TILE_SIZE
        self.rect.y = self.y * TILE_SIZE
        self.has_moved = False

    def is_clicked(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

    def get_move_range(self, move_range=3):
        """ 移動可能範囲を返す """
        move_positions = []
        for dx in range(-move_range, move_range + 1):
            for dy in range(-move_range, move_range + 1):
                if abs(dx) + abs(dy) <= move_range:
                    move_positions.append((self.x + dx, self.y + dy))
        return move_positions

    """def get_attack_range(self, move_range=3):
        # 攻撃可能範囲を返す（移動範囲+1）
        attack_range = move_range + 1
        attack_positions = []
        for dx in range(-attack_range, attack_range + 1):
            for dy in range(-attack_range, attack_range + 1):
                if abs(dx) + abs(dy) <= attack_range:
                    attack_positions.append((self.x + dx, self.y + dy))
        return attack_positions"""

    def get_attack_range(self, attack_range=1):
        """ 攻撃可能範囲を返す """
        attack_positions = []
        for dx in range(-attack_range, attack_range + 1):
            for dy in range(-attack_range, attack_range + 1):
                if abs(dx) + abs(dy) <= attack_range:
                    attack_positions.append((self.x + dx, self.y + dy))
        return attack_positions

    def attack(self, target):
        """ 対象に攻撃を行う """
        # ここでダメージ計算やヒット/ミスのロジックを実装
        #print(f"{self.name} attacks {target.name}!")
        print(" attack!")


class Enemy(Character):
    def __init__(self, x, y, img_path):
        super().__init__(x, y, img_path)

    def random_move(self, grid_width, grid_height):
        # ランダムな方向を選択（上、下、左、右）
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        new_x = max(0, min(self.x + dx, grid_width - 1))
        new_y = max(0, min(self.y + dy, grid_height - 1))
        self.move(new_x - self.x, new_y - self.y)