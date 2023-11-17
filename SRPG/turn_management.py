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
