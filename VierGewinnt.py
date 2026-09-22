import numpy

class VierGewinnt:
    """ VierGewinnt ist die Lernumgebung bzw. das eigentliche Spiel mit der gesamten Logik """
    def __init__(self):
        self.board : numpy.ndarray = numpy.zeros((6, 7))
        self.current_player : int = int(numpy.random.choice([-1, 1]))
        self.done : bool = False
        self.outcome : str | None = None
        self.players : dict = {
            1 : "Spieler 1",
            -1 : "Spieler 2"
        }

    def reset(self):
        """ Setzt alle Werte der Umgebung zurück auf Standard """
        self.board = numpy.zeros((6, 7))
        self.current_player = int(numpy.random.choice([-1, 1]))
        self.done = False
        self.outcome = None

    def get_state(self) -> numpy.ndarray:
        """ Gibt Kopie des Spielfelds aus """
        return self.board.copy()

    def get_state_str(self) -> str:
        """ Gibt Spielfeld visualisiert als String aus """
        players : dict = {
            numpy.float64(1.0): " input_data ",
            numpy.float64(-1.0): " O ",
            numpy.float64(0.0): " . "
        }
        board_str : str = ""

        for row in range(6):
            row_str : str = ""

            for col in range(7):
                row_str += players[self.board[row][col]]
            board_str += f"{6 - row} {row_str}\n"

        board_str += f"   1  2  3  4  5  6  7\n"

        return board_str

    def get_valid_moves(self) -> list[int]:
        """ Gibt alle legalen Züge aus """
        return [c for c in range(7) if self.board[0, c] == 0]

    def check_win(self, tile) -> bool:
        """ Überprüft, ob durch den aktuellen Zug ein Sieg vorhanden ist """
        row : int = tile[0]
        col : int = tile[1]
        directions : list[tuple] = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count : int = 1

            for delta in [1, -1]:
                r, c = row + dr * delta, col + dc * delta

                while 0 <= r < 6 and 0 <= c < 7 and self.board[r, c] == self.current_player:
                    count += 1
                    r += dr * delta
                    c += dc * delta

            if count >= 4:
                return True

        return False

    def check_draw(self) -> bool:
        """ Überprüft, ob unentschieden vorliegt """
        if not 0 in self.get_state():
            self.done = True
            return True
        else:
            return False

    def random_move(self) -> int:
        """ Gibt zufälligen, legalen Zug aus """
        valid_moves : list[int] = self.get_valid_moves()
        action : int = numpy.random.choice(valid_moves)

        return action

    def make_move(self, col : int) -> tuple[int, int] | None:
        """ Führt Zug auf dem Spielfeld aus """
        for row in list(reversed(range(6))):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                return row, col

        return None

    def step(self, action):
        """ Kompletter Zug inklusive Siegesüberprüfung etc.  """
        move : tuple[int, int] = self.make_move(action)
        won : bool = self.check_win(move)

        if won:
            self.outcome = "win"
            self.done = True

        elif self.check_draw():
            self.outcome = "draw"
            self.done = True

    def show_board(self):
        """ Druckt Spielzustand """
        print(self.get_state_str())
