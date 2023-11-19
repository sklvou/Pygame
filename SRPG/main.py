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
    Character(2, 2, 'img/character/brave.png', 'Yuusya A'),
    Character(3, 3, 'img/character/brave.png', 'Yuusya B'),
]
enemies = [
    Enemy(5, 5, 'img/character/skeleton.png', 'Skeleton A'),
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
enemy_selection_menu = None


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
                    # 選択したオプションが有効な場合にのみ処理
                    if menu_selection in menu.enabled_options:
                        if menu_selection == "Attack":
                            # 攻撃範囲内のエネミーを取得
                            attackable_enemies = [
                                enemy for enemy in enemies
                                if (enemy.x, enemy.y) in selected_character.get_attack_range()
                            ]
                            print(attackable_enemies)
                            # 攻撃可能なエネミーの名前でポップアップメニューを作成
                            if attackable_enemies:
                                enemy_names = [enemy.name for enemy in attackable_enemies]
                                print(enemy_names)
                                enemy_selection_menu = PopupMenu(selected_character.x * TILE_SIZE, selected_character.y * TILE_SIZE, enemy_names)
                                show_menu = False

                        elif menu_selection == "State":
                            # 待機の処理
                            selected_character.has_moved = True  # キャラクターの行動を完了とする
                            selected_character = None  # 選択を解除する

                        elif menu_selection == "Cancel":
                            # キャンセルの処理
                            if selected_character:
                                selected_character.cancel_movement()
                            selected_character = None  # 選択を解除する
                        show_menu = False
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
                        #if player.is_clicked(mouse_pos) and not player.has_moved:
                            selected_character = player
                            player.initial_x = player.x
                            player.initial_y = player.y
                            move_range = player.get_move_range()
                            attack_range = player.get_attack_range()
                            character_has_moved = False  # 移動していない状態にリセット

                        # キャラクターが選択されていて、移動範囲内がクリックされた場合
                        elif selected_character == player and grid_pos in move_range:
                            selected_character.move(grid_pos[0] - selected_character.x, grid_pos[1] - selected_character.y)
                            move_range = []
                            attack_range = []
                            character_has_moved = True  # キャラクターが移動した

                # 移動が完了した後にメニューを表示
                if character_has_moved and not show_menu and selected_character:
                    # 攻撃範囲内のエネミーを取得
                    attackable_enemies = [
                        enemy for enemy in enemies
                        if (enemy.x, enemy.y) in selected_character.get_attack_range()
                    ]
                    # 攻撃可能なエネミーがいれば攻撃オプションを有効化
                    enabled_options = ["State", "Cancel"]
                    if attackable_enemies:
                        enabled_options.insert(0, "Attack")  # 攻撃オプションを追加

                    menu = PopupMenu(selected_character.x * TILE_SIZE, selected_character.y * TILE_SIZE, ["Attack", "State", "Cancel"], enabled_options)
                    show_menu = True
                    character_has_moved = False  #  メニュー表示後はフラグをリセット


            if enemy_selection_menu:
                # ユーザーのクリックイベントを待つ
                if event.type == pygame.MOUSEBUTTONDOWN:
                    selected_enemy_name = enemy_selection_menu.get_selection(pygame.mouse.get_pos())
                    if selected_enemy_name:
                        # 選択されたエネミーを取得
                        selected_enemy = next((enemy for enemy in attackable_enemies if enemy.name == selected_enemy_name), None)
                        if selected_enemy:
                            # 戦闘シーンに遷移する処理...
                            print(f"Entering battle with {selected_enemy.name}")
                            # エネミー選択メニューを閉じる
                            enemy_selection_menu = None
                            # 戦闘シーンへの遷移処理（擬似コード）
                            selected_character.attack(selected_enemy)
                            selected_character.has_moved = True  # キャラクターの行動を完了とする
                            selected_character = None  # 選択を解除する

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

    if enemy_selection_menu:
        enemy_selection_menu.draw(screen)

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