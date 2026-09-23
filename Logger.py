import pandas
import numpy as np
import matplotlib.pyplot as plt


class Logger:
    """ Verwaltet das Speichern und Darstellen der Trainingsdaten. """
    def __init__(self, directory : str):
        self.log_data_a : list[tuple] = []
        self.log_data_b : list[tuple] = []

        self.directory : str = directory

        self.last_win_rate_a : float = 0.0
        self.last_win_rate_b : float = 0.0

    """ Gilt für alle (save_[...]) Methoden: Session statt einzelner Variablen wie current_episode als Parameter, um Anzahl 
    an Parametern zu verringern, Methoden waren ursprünglich Teil der Session Klasse, aber verschoben für Einheitlichkeit. """
    def save_checkpoint(self, session, step):
        """ Speichert Modelle und Replays alle n steps """
        current_episode : int = session.current_episode_number
        total_episodes : int = session.total_episodes

        if current_episode % step == 0:
            print(f"Episode: {current_episode}/{total_episodes}, Score: {session.agent_a.reward}, Epsilon: {session.agent_a.epsilon:.2f}")
            self.save_replay(session)

            if session.agent_b:
                session.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_a_episode{current_episode}.pth")
                session.agent_b.save_model(f"training/{self.directory}/checkpoints/agent_b_episode{current_episode}.pth")
            else:
                session.agent_a.save_model(f"training/{self.directory}/checkpoints/agent_episode{current_episode}.pth")

    """ Um besser nachzuvollziehen was genau der Agent im Training macht und mögliche Fehler schneller erfassen zu können """
    def save_replay(self, session):
        """ Speichert den Spielstand als txt Datei. """
        current_episode : int = session.current_episode_number

        with open(f"training/{self.directory}/checkpoints/game_episode{current_episode}.txt", "x") as file:
            if session.agent_pool:
                file.write(f"Anzahl Agenten im Agentenpool: {len(session.agent_pool)}\n Aktueller Agent: {session.agent_pool_index} \n")

            for state in session.current_game:
                file.write(f"{state}\n")

    def save_episode(self, session):
        """ Speichert Checkpoints für Episoden mit variierender Schrittweite.  """
        current_episode : int = session.current_episode_number
        total_episodes : int = session.total_episodes

        if current_episode < 1000:
            self.save_checkpoint(session, 100)
        else:
            self.save_checkpoint(session, 1000)

        if current_episode + 1 == total_episodes:
            session.agent_a.save_model(f"training/{self.directory}/final_agent_a_episode{total_episodes}.pth")
            if session.agent_b:
                session.agent_b.save_model(f"training/{self.directory}/final_agent_b_episode{total_episodes}.pth")

    def log_episode(self, session, new_win_rate_a=None, new_win_rate_b=None):
        """ Erweitert das Log um einen weiteren Datensatz """
        if new_win_rate_a:
            self.last_win_rate_a = new_win_rate_a

        self.log_data_a.append((
            session.current_episode_number,
            session.agent_a.reward,
            session.agent_a.epsilon,
            session.agent_a.current_loss,
            self.last_win_rate_a
        ))

        if session.agent_b:
            if new_win_rate_b:
                self.last_win_rate_b = new_win_rate_b

            self.log_data_b.append((
                session.current_episode_number,
                session.agent_b.reward,
                session.agent_b.epsilon,
                session.agent_b.current_loss,
                self.last_win_rate_b
            ))

    def save_logs_to_csv(self):
        """ Speichert Logs als CSV-Datei """
        columns : list[str] = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]
        data_frame_a : pandas.DataFrame = pandas.DataFrame(self.log_data_a, columns=columns)

        if len(self.log_data_b):
            data_frame_a.to_csv(f"training/{self.directory}/training_log_a.csv", index=False)

            data_frame_b : pandas.DataFrame = pandas.DataFrame(self.log_data_b, columns=columns)
            data_frame_b.to_csv(f"training/{self.directory}/training_log_b.csv", index=False)
        else:
            data_frame_a.to_csv(f"training/{self.directory}/training_log.csv", index=False)

        print("Trainingssession beendet und Log gespeichert.")

    @staticmethod
    def plot_loss(data_frame, shape = (2, 2), location = (0, 0)):
        """ Plottet alle Loss-Datenpunkte, sowie ein geglätteter Graph und verdeutlicht Datenpunkte mit mehr als 3-Facher Standardabweichung """
        ax : plt.Axes = plt.subplot2grid(shape, location, xlabel="Episode", ylabel="Loss", title="Loss Graph")
        ax.plot(data_frame["Loss"],
                'o',
                markersize = 2,
                color = 'navy',
                alpha = 1000 / len(data_frame) if len(data_frame) > 1000 else 1,
                label = 'Loss-Events'
                )

        mean = data_frame["Loss"].mean()
        std = data_frame["Loss"].std()
        threshold = mean + 3 * std

        outliers = data_frame[data_frame["Loss"] > threshold]
        ax.scatter(outliers.index,
                   outliers["Loss"],
                   s = 10,
                   color = "crimson",
                   alpha = 10000/len(data_frame) if len(data_frame) > 10000 else 1,
                   label = 'Kritische Ausreißer'
                   )

        smoothed_loss = data_frame["Loss"].rolling(window = 100).mean()
        ax.plot(smoothed_loss,
                color = "navy",
                label = "Loss Smooth"
                )

        ax.legend()

    @staticmethod
    def plot_win_rate(data_frame, shape = (2, 2), location = (1, 0)):
        """ Plottet Siegquote als Graph sowie eine Durchschnittsgerade. """
        ax : plt.Axes = plt.subplot2grid(shape, location, xlabel = "Episode", ylabel = "WinRate", ylim = (0, 1), title="Win Rate Graph")

        # Nur jeden 100. Datenpunkt plotten, da nur alle 100 Episoden WinRate berechnet wird
        ax.plot(data_frame["WinRate"].iloc[::100],
                linestyle="--",
                marker="o",
                markersize=5,
                color = "coral",
                alpha = 0.5,
                label = "WinRate"
                )

        # Durchschnittsgerade
        x = np.arange(len(data_frame))
        y = data_frame["WinRate"].values
        m, b = np.polyfit(x, y, 1)
        ax.plot(x,
                m * x + b,
                color = "coral",
                label = "Trend"
                )

        ax.legend()

    @staticmethod
    def plot_reward(data_frame, shape = (2, 2), location = (0, 1), row = 2, epsilon = False):
        """ Plottet alle Datenpunkte der gesamten Belohnung am Ende einer Episode sowie einen geglätteten Graphen,
        eine Durchschnittsgerade und im Overview auch Epsilon als Graphen"""
        ax : plt.Axes = plt.subplot2grid(shape, location, rowspan = row, xlabel="Episode", ylabel="Reward", title="Reward Graph")

        ax.plot(data_frame["Reward"],
                "o",
                markersize = 2,
                color = "teal",
                alpha = 0.125,
                label = "Reward-Events"
                )

        x : np.ndarray = np.arange(len(data_frame))
        y : pandas.DataFrame = data_frame["Reward"].values
        m, b = np.polyfit(x, y, 1)
        ax.plot(x,
                m * x + b,
                color = "teal",
                label = "Trend"
                )

        smoothed_loss = data_frame["Reward"].rolling(window = 100).mean()
        ax.plot(smoothed_loss,
                color = "teal",
                alpha = 0.5,
                label = "Reward Smooth"
                )

        if epsilon:
            ax.plot(data_frame["Epsilon"],
                    color = "crimson",
                    label = "Epsilon"
                    )

        ax.legend()

    """ Reward als "höchste" Statistik, weil es hier die größten ausschweifungen in Y-Richtung gibt. """
    def plot(self, data_frame = None, name = ""):
        """ Plottet Belohnungen, Siegquote und Loss für eine Session"""
        plt.figure(figsize = (20, 10))

        if data_frame is None:
            columns = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]
            if name == "b":
                data_frame = pandas.DataFrame(self.log_data_b, columns=columns)
            else:
                data_frame = pandas.DataFrame(self.log_data_a, columns=columns)

        self.plot_loss(data_frame)
        self.plot_win_rate(data_frame)
        self.plot_reward(data_frame)

        if name != "":
            name = f"_{name}"

        plt.tight_layout()
        plt.savefig(f"training/{self.directory}/training_log{name}.svg", format="svg")
        plt.savefig(f"training/{self.directory}/training_log{name}.png", format="png")

        plt.close()

    """ eher länglich, da es deutlich mehr Episoden gibt und es weniger wichtig ist genaue Punkte ablesen zu können, als 
    einen groben Überblick zu erhalten """
    def plot_overview(self, data_frame = None, name = ""):
        """ Plottet Belohnungen, Siegquote, Loss und Epsilon als Überblick für mehrere Sessions """
        plt.figure(figsize=(20, 10))

        self.plot_loss(data_frame, shape = (4, 1), location = (0, 0))
        self.plot_win_rate(data_frame, shape = (4, 1), location = (1, 0))
        self.plot_reward(data_frame, shape = (4, 1), location = (2, 0), epsilon=True)

        if name != "":
            name = f"_{name}"

        plt.tight_layout()

        plt.savefig(f"training/{self.directory}/training_overview{name}.svg", format="svg")
        plt.savefig(f"training/{self.directory}/training_overview{name}.png", format="png", facecolor='white', transparent=False)

        plt.close()

    def plot_comparison(self, data_a : pandas.DataFrame, data_b : pandas.DataFrame):
        plt.figure(figsize=(20, 40))  # auch Höhe anpassen

        self.plot_loss(data_a, shape=(8, 1), location=(0, 0))
        self.plot_loss(data_b, shape=(8, 1), location=(1, 0))

        self.plot_win_rate(data_a, shape=(8, 1), location=(2, 0))
        self.plot_win_rate(data_b, shape=(8, 1), location=(3, 0))

        self.plot_reward(data_a, shape=(8, 1), location=(4, 0), epsilon=True)
        self.plot_reward(data_b, shape=(8, 1), location=(6, 0), epsilon=True)

        plt.tight_layout()

        plt.savefig(f"training/{self.directory}/training_comparison.svg", format="svg")
        plt.savefig(f"training/{self.directory}/training_comparison.png", format="png", facecolor='white', transparent=False)

        plt.close()

if __name__ == "__main__":
    """logger = Logger(directory="trainer_26/base_model")

    old_data = pandas.read_csv("training/trainer_26/base_model/training_log.csv")

    data = [pandas.read_csv("training/trainer_26/base_model/training_log.csv"),
            pandas.read_csv("training/trainer_26/league_play_model/training_log.csv"),
            pandas.read_csv("training/trainer_26/league_play_model_1/training_log.csv")
            ]

    new_data = pandas.concat(data, ignore_index=True)



    #logger.plot(data)
    logger.plot_overview(new_data)
    """
    logger = Logger(directory="trainer_34/self_play_model")
    columns = ["Episode", "Reward", "Epsilon", "Loss", "WinRate"]

    data_a = pandas.read_csv("training/trainer_34/self_play_model/training_log_a.csv")
    #logger.plot(data_a, "a")

    data_b = pandas.read_csv("training/trainer_34/self_play_model/training_log_b.csv")
    #logger.plot(data_b, "b")

    logger.plot_comparison(data_a, data_b)
