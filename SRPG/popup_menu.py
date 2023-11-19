import pygame

class PopupMenu:
    def __init__(self, x, y, menu_options, enabled_options=None):
        self.x = x
        self.y = y
        self.menu_options = menu_options
        self.enabled_options = enabled_options if enabled_options is not None else menu_options # 選択可能かどうか
        self.width = 200  # メニューの幅を設定
        self.height = len(menu_options) * 40  # メニューの高さを設定（各項目の高さ40pxと仮定）
        self.font = pygame.font.Font(None, 32)
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)  # メニューの矩形領域を作成


    def draw(self, screen):
        # メニュー背景
        pygame.draw.rect(screen, (255, 255, 255), self.rect)

        # メニューオプション
        for i, text in enumerate(self.menu_options):
            if text in self.enabled_options:
                text_color = (0, 0, 0)  # 有効オプションの色
            else:
                text_color = (200, 200, 200)  # 無効オプションの色
            text_surf = self.font.render(text, True, text_color)
            screen.blit(text_surf, (self.x + 10, self.y + i * 40 + 10))
            
    def get_selection(self, click_position):
        if self.rect.collidepoint(click_position):
            # クリックされたメニュー項目を判断するロジックをここに実装
            item_height = self.rect.height // len(self.menu_options)
            index = (click_position[1] - self.rect.y) // item_height
            if index < len(self.menu_options):
                return self.menu_options[index]
        return None