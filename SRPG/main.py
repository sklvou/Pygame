import pygame
import random
import sys
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
while running:
    # イベント処理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # プレイヤーターン
        if turn_manager.is_player_turn:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                grid_pos = mouse_pos[0] // TILE_SIZE, mouse_pos[1] // TILE_SIZE
                
                for player in players:
                    # キャラクター選択
                    if selected_character is None and player.is_clicked(mouse_pos) and not player.has_moved:
                        selected_character = player
                        move_range = player.get_move_range()
                        attack_range = player.get_attack_range()

                    # 移動先選択
                    elif selected_character == player and grid_pos in move_range:
                        selected_character.move(grid_pos[0] - selected_character.x, grid_pos[1] - selected_character.y)
                        selected_character = None
                        move_range = []
                        attack_range = []

                    # 攻撃処理
                    if selected_character and event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()
                        grid_pos = mouse_pos[0] // TILE_SIZE, mouse_pos[1] // TILE_SIZE

                        # 攻撃範囲のチェック
                        attack_range = selected_character.get_attack_range()
                        if grid_pos in attack_range:
                            # 攻撃対象の選定と攻撃の実行
                            for enemy in enemies:
                                if (enemy.x, enemy.y) == grid_pos:
                                    selected_character.attack(enemy)
                                    break


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

    # 範囲の可視化
    if selected_character:
        for pos in attack_range:
            pygame.draw.rect(screen, (255, 0, 0), (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)
        for pos in move_range:
            pygame.draw.rect(screen, (0, 255, 0), (pos[0] * TILE_SIZE, pos[1] * TILE_SIZE, TILE_SIZE, TILE_SIZE), 2)

    pygame.display.flip()

    # FPSを制御
    clock.tick(FPS)

# Pygameの終了処理
pygame.quit()
sys.exit()