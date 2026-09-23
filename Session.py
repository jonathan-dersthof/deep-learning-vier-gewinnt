import numpy
import pandas

from VierGewinnt import VierGewinnt
from Agent import Agent
from Episode import Episode
from Logger import Logger


class Session:
    """ eine Session ist ein Trainingsablauf, sprich eine Reihe an beliebig vielen Trainingsspielen, welche unterschiedlich vielen Agenten benutzt werden kann """
    def __init__(self,
                 directory : str,
                 total_episodes : int,
                 agent_a : Agent,
                 agent_b : Agent = None,
                 agent_pool : list[Agent] = None
                 ):
        # Agent A wird prinzipiell immer als lernender Agent behandelt
        self.agent_a : Agent = agent_a
        # Agent B wird im fall Self Play auch als lernend behandelt
        self.agent_b : Agent | None = agent_b
        # Agentenpool lernt nicht
        self.agent_pool : list[Agent] = agent_pool
        self.agent_pool_index : int = 0

        self.env : VierGewinnt = VierGewinnt()
        self.directory : str = directory

        self.total_episodes : int = total_episodes
        self.current_episode_number : int = 0

        self.current_game : list[str] = []

        self.logger : Logger = Logger(directory)

    def learn(self):
        """ lässt alle lernenden Agenten der Session lernen. """
        self.agent_a.replay()

        if self.agent_b:
            self.agent_b.replay()

    def evaluate_agent(self, agent : Agent, num_test_games : int = 100) -> float:
        """ simuliert 100 Testspiele gegen einen Zufallsgegner, um als Benchmark für den Fortschritt zu dienen. """
        wins : int = 0
        draws : int = 0
        losses : int = 0

        saved_epsilon : float = agent.epsilon
        agent.epsilon = 0.0

        for i in range(num_test_games):
            self.env.reset()
            self.env.current_player = 1

            while not self.env.done:
                if self.env.current_player == 1:
                    action : int = agent.act(self.env)
                else:
                    action : int = self.env.random_move()

                self.env.step(action)

                if not self.env.done:
                    self.env.current_player *= -1

            if self.env.outcome == "win":
                if self.env.current_player == 1:
                    wins += 1
                else:
                    losses += 1
            elif self.env.outcome == "draw":
                draws += 1

        agent.epsilon = saved_epsilon

        win_rate : float = (wins + 0.5 * draws) / num_test_games
        self.env.reset()

        return win_rate

    def run_episode(self):
        """ führt eine einzige Episode komplett aus inklusive Speichern, lernen etc """
        if self.agent_b:
            episode : Episode = Episode(self.env, self.agent_a, self.agent_b)
        # 10% Wahrscheinlichkeit gegen einen Zufallsgegner zu spielen, obwohl es einen Agentenpool gibt, um Overfitting zu vermindern
        elif self.agent_pool and numpy.random.rand() > 0.1:
            self.agent_pool_index = numpy.random.randint(len(self.agent_pool))
            episode : Episode = Episode(self.env, self.agent_a, self.agent_pool[self.agent_pool_index])
        else:
            episode : Episode = Episode(self.env, self.agent_a)

        episode.run()
        self.current_game = episode.game_states_str

        self.learn()

        new_win_rate_a : float | None = None
        new_win_rate_b : float | None = None

        if self.current_episode_number % 100 == 0:
            new_win_rate_a : float = self.evaluate_agent(self.agent_a)

            if self.agent_b:
                new_win_rate_b : float = self.evaluate_agent(self.agent_b)

        self.logger.log_episode(self, new_win_rate_a, new_win_rate_b)

        self.logger.save_episode(self)
        self.env.reset()

    def run(self):
        """ führt die gesamte Session aus """
        while self.current_episode_number < self.total_episodes:
            self.run_episode()
            self.current_episode_number += 1

        self.logger.save_logs_to_csv()

        columns = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]
        data_a = pandas.DataFrame(self.logger.log_data_a, columns = columns)

        if self.agent_b:
            data_b = pandas.DataFrame(self.logger.log_data_b, columns = columns)

            self.logger.plot(data_a, "a")
            self.logger.plot(data_b, "b")

            self.logger.plot_comparison(data_a, data_b)
        else:
            self.logger.plot(data_a)
