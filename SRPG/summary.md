main.py
```py
import pygame
import random
import sys
from popup_menu import PopupMenu
from settings import SCREEN_WIDTH, SCREEN_HEIGHT, TILE_SIZE
from map import Map
from character import Character, Enemy
from turn_management import TurnManager

# 乱数のシード値を固定
random.seed(0)

# Pygameの初期化
pygame.init()

# 画面の設定
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# FPSの設定
FPS = 60
clock = pygame.time.Clock()

# マップサイズの定義
grid_width = SCREEN_WIDTH // TILE_SIZE
grid_height = SCREEN_HEIGHT // TILE_SIZE

# ゲームの初期化
game_map = Map(grid_width, grid_height)

is_player_turn = True  # プレイヤーターンであるかどうかのフラグ

# キャラクターのインスタンスを作成
players = [
    Character(2, 2, 'img/character/brave.png'),
    Character(3, 3, 'img/character/brave.png')  # 追加のキャラクター
]
enemies = [
    Enemy(5, 5, 'img/character/skeleton.png'),
    # 他のエネミーキャラクターもここに追加
]

# キャラクターの選択状態を追跡する変数
selected_character = None
move_range = []

# ターンマネージャの初期化
turn_manager = TurnManager(players, enemies)

# ゲームループ
running = True
show_menu = False
menu = None
selected_character = None
character_has_moved = False  # キャラクターが移動したかどうかのフラグを追加

while running:
    # イベント処理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # メニュー表示中は他の選択をブロック
        if show_menu:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if menu.rect.collidepoint(mouse_pos):
                    menu_selection = menu.get_selection(mouse_pos)
                    # メニューの選択処理
                    if menu_selection == "攻撃":
                        # 攻撃の処理
                        pass
                    elif menu_selection == "待機":
                        # 待機の処理
                        pass
                    elif menu_selection == "キャンセル":
                        # キャンセルの処理
                        pass
                    show_menu = False
                    selected_character.has_moved = True  # キャラクターの行動を完了とする
                    selected_character = None  # 選択を解除する
                continue  # メニュー表示中は他の処理をスキップ

        # プレイヤーターン
        if turn_manager.is_player_turn:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                grid_pos = mouse_pos[0] // TILE_SIZE, mouse_pos[1] // TILE_SIZE

                if not show_menu: # メニューが表示されていない場合のみキャラクター選択を処理
                    for player in players:
                        # キャラクター選択
                        if selected_character is None and player.is_clicked(mouse_pos) and not player.has_moved:
                            selected_character = player
                            move_range = player.get_move_range()
                            attack_range = player.get_attack_range()
                            #character_has_moved = False  # 移動していない状態にリセット

                        # キャラクターが選択されていて、移動範囲内がクリックされた場合
                        elif selected_character == player and grid_pos in move_range:
                            selected_character.move(grid_pos[0] - selected_character.x, grid_pos[1] - selected_character.y)
                            move_range = []
                            attack_range = []
                            character_has_moved = True  # キャラクターが移動した

                        """# 攻撃処理
                        if selected_character and event.type == pygame.MOUSEBUTTONDOWN:
                            mouse_pos = pygame.mouse.get_pos()
                            grid_pos = mouse_pos[0] // TILE_SIZE, mouse_pos[1] // TILE_SIZE

                            # 攻撃範囲のチェック
                            if grid_pos in attack_range:
                                # 攻撃対象の選定と攻撃の実行
                                for enemy in enemies:
                                    if (enemy.x, enemy.y) == grid_pos:
                                        selected_character.attack(enemy)
                                        break"""

                # 移動が完了した後にメニューを表示
                if character_has_moved and not show_menu and selected_character:
                    menu = PopupMenu(selected_character.rect.x + TILE_SIZE, selected_character.rect.y, ["Attack", "State", "Cancel"])
                    show_menu = True
                    character_has_moved = False  #  メニュー表示後はフラグをリセット

            
            # ターン自動終了のチェック
            if turn_manager.check_turn_end():
                turn_manager.end_turn()
            
        # エネミーターン
        else:
            # エネミーの行動
            for enemy in enemies:
                if not enemy.has_moved:
                    enemy.random_move(grid_width, grid_height)
                    enemy.has_moved = True

            # ターン自動終了のチェック
            if turn_manager.check_turn_end():
                turn_manager.end_turn()

    # 画面の描画
    screen.fill((0, 0, 0))
    game_map.draw(screen)
    for player in players:
        player.draw(screen)  # プレイヤーキャラクターを描画
    for enemy in enemies:
        enemy.draw(screen)  # エネミーキャラクターを描画
    if show_menu and menu:
        menu.draw(screen)  # ポップアップメニューを描画

    # 範囲の可視化
    if selected_character:
        for pos in move_range:
            pygame.draw.rect(screen, (0, 255, 0), (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)
        for pos in attack_range:
            pygame.draw.rect(screen, (255, 0, 0), (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)

    pygame.display.flip()

    # FPSを制御
    clock.tick(FPS)

# Pygameの終了処理
pygame.quit()
sys.exit()
```

setting.py
```py
# 画面とタイルの設定
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
TILE_SIZE = 40
```

map.py
```py
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
```

character.py
```py
import random
import pygame
from settings import TILE_SIZE
from map import load_and_scale_img

class Character:
    def __init__(self, x, y, img_path):
        self.x = x
        self.y = y
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
```

popup_menu.py
```py
import pygame

class PopupMenu:
    def __init__(self, x, y, menu_options):
        self.x = x
        self.y = y
        self.menu_options = menu_options
        self.width = 200  # メニューの幅を設定
        self.height = len(menu_options) * 40  # メニューの高さを設定（各項目の高さ40pxと仮定）
        self.font = pygame.font.Font(None, 32)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)  # メニューの矩形領域を作成

    def draw(self, screen):
        # メニュー背景
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

        # メニューオプション
        for i, text in enumerate(self.menu_options):
            text_surf = self.font.render(text, True, (0, 0, 0))
            screen.blit(text_surf, (self.x + 10, self.y + i * 40 + 10))

    def get_selection(self, click_position):
        if self.rect.collidepoint(click_position):
            # クリックされたメニュー項目を判断するロジックをここに実装
            item_height = self.rect.height // len(self.menu_options)
            index = (click_position[1] - self.rect.y) // item_height
            if index < len(self.menu_options):
                return self.menu_options[index]
        return None
```

turn_management.py
```py
class TurnManager:
    def __init__(self, players, enemies):
        self.players = players
        self.enemies = enemies
        self.is_player_turn = True

    def start_turn(self):
        self.reset_movements(self.current_team())

    def end_turn(self):
        self.is_player_turn = not self.is_player_turn
        self.start_turn()

    def current_team(self):
        return self.players if self.is_player_turn else self.enemies

    def reset_movements(self, characters):
        for char in characters:
            char.reset_movement()

    def check_turn_end(self):
        return all(char.has_moved for char in self.current_team())
```