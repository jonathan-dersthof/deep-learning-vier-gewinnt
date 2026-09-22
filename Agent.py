import random
import numpy
import pickle

import NumpyImplementation as npImp

from VierGewinnt import VierGewinnt
from collections import deque


class Agent:
    """ Die Agent-Klasse verwaltet das DQN Modell und beinhaltet alle Methoden für das Lernen und Handeln des Agenten """
    def __init__(self,
                 hidden_size : int = 256,

                 gamma : float = 0.95,
                 epsilon : float = 1.0,
                 epsilon_min : float = 0.01,
                 epsilon_decay : float = 0.99995,
                 learning_rate : float = 0.001,
                 batch_size : int = 64,

                 winner_reward : float = 1.0,
                 draw_reward : float = 0.1,
                 lose_reward : float = -1.0,
                 survive_reward : float = 0.01
                 ):
        self.hidden_size : int = hidden_size

        self.memory : deque = deque(maxlen = 100000)

        self.gamma : float= gamma
        self.epsilon : float = epsilon
        self.epsilon_min : float = epsilon_min
        self.epsilon_decay : float = epsilon_decay
        self.learning_rate : float = learning_rate
        self.batch_size : int = batch_size

        # PyTorch durch NumPy ersetzt. Input = 42 (6x7 Feld), Output = 7 (Spalten)
        self.model : npImp.LinearDQN = npImp.LinearDQN(
            input_size = 42,
            output_size = 7,
            hidden_size = hidden_size,
            hidden_layers=  2
        )
        self.target_model : npImp.LinearDQN = npImp.LinearDQN(
            input_size = 42,
            output_size = 7,
            hidden_size = hidden_size,
            hidden_layers = 2
        )
        self.update_target_network()

        # Schrittweite in welcher target_model aktualisiert wird
        self.update_step : int= 1000
        self.steps : int = 0

        # NumPy Optimizer und Loss
        self.optimizer : npImp.Adam = npImp.Adam(learning_rate = self.learning_rate)
        self.criterion : npImp.MSELoss = npImp.MSELoss()

        self.win_reward : float = winner_reward
        self.draw_reward : float = draw_reward
        self.lose_reward : float = lose_reward
        self.survive_reward : float = survive_reward

        # gesammelter Reward innerhalb einer Episode
        self.reward : float = 0.0
        self.current_loss : float = 0.0

    def set_hyperparameters(self,
                            gamma: float,
                            epsilon: float,
                            epsilon_min: float,
                            epsilon_decay: float,
                            learning_rate: float,
                            batch_size: int,

                            win_reward: float,
                            draw_reward: float,
                            lose_reward: float,
                            survive_reward: float
                            ):
        """ Setzt die Hyperparameter des Agenten neu"""
        self.gamma : float = gamma
        self.epsilon : float = epsilon
        self.epsilon_min : float = epsilon_min
        self.epsilon_decay : float = epsilon_decay
        self.learning_rate : float = learning_rate
        self.batch_size : int = batch_size

        self.win_reward : float = win_reward
        self.draw_reward : float = draw_reward
        self.lose_reward : float = lose_reward
        self.survive_reward : float = survive_reward

    def remember(self,
                 state : numpy.ndarray,
                 action : int,
                 reward : float,
                 next_state : numpy.ndarray,
                 done : bool):
        """ Fügt der Erinnerung des Agenten einen weiteren Datensatz hinzu. """
        self.memory.append((state, action, reward, next_state, done))

    def update_target_network(self):
        """ Kopiert die Gewichte vom Policy-Netzwerk in das Target-Netzwerk """
        for target_layer, policy_layer in zip(self.target_model.layers, self.model.layers):
            if hasattr(target_layer, 'weights'):
                target_layer.weights = policy_layer.weights.copy()
                target_layer.biases = policy_layer.biases.copy()

    def act(self, env : VierGewinnt) -> int:
        """ Die Methode wählt den nächsten Zug aus einer Liste an legalen Zügen auf dem Brett aus. """
        state : numpy.ndarray = env.get_state()

        # state Normalisierung: der Agent sieht das Spielbrett immer aus der Sicht von Spieler 1
        # → dadurch muss der Agent nicht lernen zu unterscheiden welcher Spieler er ist
        if env.current_player == -1:
            state *= -1

        valid_moves : list[int] = env.get_valid_moves()

        if not valid_moves:
            return 0

        # exploration
        if numpy.random.rand() <= self.epsilon:
            return random.choice(valid_moves)

        state_flat = state.flatten()[numpy.newaxis, :]

        act_values = self.model.forward(state_flat)
        q_values = act_values[0].copy()

        # illegale Züge erhalten einen Q-Wert in der negativen Unendlichkeit, da in Randerscheinungen im Training trotz einem vorher gewählten sehr niedrigen Wert illegale Züge gewählt wurden
        for column in range(7):
            if env.board[0, column] != 0:
                q_values[column] = -numpy.inf

        # wählt alle möglichen Aktionen mit dem gleichen Maximal vorausgesagtem Q-Wert als potenzielle Aktionen aus
        max_q : float = numpy.max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]

        return random.choice(best_actions)

    """ Um den Agent klarer vom Environment zu trennen, wurde die Reward Logik aus der VierGewinnt  Klasse nach Episode verschoben.
    Da in diesem Rahmen jeder überlebte Zug als survive_reward gewertet wird, muss nach Spielende die letzte Erinnerung überschrieben werden, um win_reward etc zu verteilen"""
    def correct_last_reward(self, reward : float):
        """ Korrigiert letzten Eintrag der Erinnerung. """
        if self.memory:
            last_memory = list(self.memory[-1])
            last_memory[2] = reward
            last_memory[4] = True
            self.memory[-1] = tuple(last_memory)
            self.reward += reward

    def replay(self):
        """ In der Methode findet die Auswertung der letzten Episode statt. """
        if len(self.memory) < self.batch_size:
            return

        minibatch : list[tuple] = random.sample(self.memory, self.batch_size)

        states = numpy.array([x[0].flatten() for x in minibatch], dtype = numpy.float32)
        actions = numpy.array([x[1] for x in minibatch], dtype = numpy.int32)
        rewards = numpy.array([x[2] for x in minibatch], dtype = numpy.float32)
        next_states = numpy.array([x[3].flatten() for x in minibatch], dtype = numpy.float32)
        dones = numpy.array([x[4] for x in minibatch], dtype = numpy.float32)

        current_q_values = self.model.forward(states)

        next_q_values = self.model.forward(next_states)
        next_actions = numpy.argmax(next_q_values, axis=1)

        # Q-Wert dieser Aktion mit dem Target-Modell bewerten
        target_next_q_values = self.target_model.forward(next_states)
        batch_indices = numpy.arange(self.batch_size)
        max_next_q_values = target_next_q_values[batch_indices, next_actions]

        # Bellman-Gleichung
        updated_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        targets = current_q_values.copy()
        targets[batch_indices, actions] = updated_q_values

        self.current_loss = self.criterion.forward(current_q_values, targets)
        d_output = self.criterion.backward()

        # 5. Gradient Clipping (ersetzt torch.nn.utils.clip_grad_norm_)
        # Verhindert explodierende Gradienten durch Kappung auf Werte zwischen -1 und 1
        d_output = numpy.clip(d_output, -1.0, 1.0)

        self.model.backward(d_output)
        self.optimizer.step(self.model)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.steps += 1

        if self.steps % self.update_step == 0:
            self.update_target_network()

    def save_model(self, path: str):
        """ Speichert das gesamte NumPy-Modell mittels Pickle """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Modell gespeichert unter {path}")

    def load_model(self, path: str):
        """ Lädt das NumPy-Modell und synchronisiert das Target-Modell """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.update_target_network()
        print(f"Modell aus '{path}' geladen.")
