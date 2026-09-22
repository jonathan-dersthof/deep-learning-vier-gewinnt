from logic import select_path, next_directory, get_attributes_from_file, select_value, select_agent
from Trainer import Trainer
from Agent import Agent


class Training:
    """ Nutzergesteuertes Training """
    def __init__(self):
        self.trainer : Trainer | None = None

    def setup(self):
        """ Lässt alten Trainer auswählen oder neuen Erstellen """
        while True:
            user_input : str = input(f"Trainer wählen oder neuen Trainer erstellen?:\n1. Trainer Wählen\n2. Neuer Trainer\n> ")

            if user_input == "1" or "wählen" in user_input.lower():
                selected_trainer = select_path("training", "trainer_")
                hidden_size = get_attributes_from_file(f"{selected_trainer}/DQN_attributes")["hidden size"]
                break
            elif user_input == "2" or "neu" in user_input.lower():
                selected_trainer = next_directory("training", "trainer")
                hidden_size = select_value("hidden size wählen", 256, "int")
                break
            else:
                print("Option 1 oder 2 wählen.")

        self.trainer = Trainer(hidden_size = hidden_size,
                               directory = selected_trainer
                               )

    @staticmethod
    def get_hyperparameters(extension = "") -> dict:
        """ Lässt alle Hyperparameter für Agenten festlegen und gibt Standardwerte als Referenz an """
        default_agent : Agent = Agent()

        print("Hyperparameter festlegen: ")
        gamma : float = select_value("Gamma", default = default_agent.gamma, value_type = "float")
        epsilon : float = select_value("Epsilon", default = default_agent.epsilon, value_type = "float")
        epsilon_min : float = select_value("Minimalwert für Epsilon (epsilon_min)", default = default_agent.epsilon_min, value_type = "float")
        epsilon_decay : float = select_value("Epsilon Verfallsrate (epsilon_decay)", default = default_agent.epsilon_decay, value_type = "float")
        learning_rate : float = select_value("Lernrate (learning_rate)", default = default_agent.learning_rate, value_type = "float")
        batch_size : int = select_value("Losgröße (batch_size)", default = default_agent.batch_size, value_type = "int")

        print("Belohnungen festlegen für: ")
        win_reward : float  = select_value("Sieg(win_reward)", default = default_agent.win_reward, value_type = "float")
        draw_reward : float = select_value("Unentschieden (draw_reward)", default = default_agent.draw_reward, value_type = "float")
        lose_reward : float = select_value("Niederlage (lose_reward)", default = default_agent.lose_reward, value_type = "float")
        survive_reward : float = select_value("Überleben (survive_reward)", default = default_agent.survive_reward, value_type = "float")

        if extension != "":
            extension : str = "_" + extension
        return {
            f"gamma{extension}" : gamma,
            f"epsilon{extension}" : epsilon,
            f"epsilon_min{extension}" : epsilon_min,
            f"epsilon_decay{extension}" : epsilon_decay,
            f"learning_rate{extension}" : learning_rate,
            f"batch_size{extension}" : batch_size,
            f"win_reward{extension}" : win_reward,
            f"draw_reward{extension}" : draw_reward,
            f"lose_reward{extension}" : lose_reward,
            f"survive_reward{extension}" : survive_reward
            }

    @staticmethod
    def select_training() -> str :
        """ Lässt Nutzer Trainingsart festlegen"""
        while True:
            while True:
                user_input: str = input(f"Training Wählen:\n1. Basistraining\n2. Self-play\n3. League-play\n4. Volles Training\n> ")

                if user_input == "1" or "basis" in user_input.lower():
                    return "base_training"
                elif user_input == "2" or "self" in user_input.lower():
                    return "self_play"
                elif user_input == "3" or "league" in user_input.lower():
                    return "league_play"
                elif user_input == "4" or "voll" in user_input.lower():
                    return "full_training"
                else:
                    print("Option 1 - 4 wählen.")

    def train(self):
        """ Gesamter Ablauf des Nutzergesteuerten Trainings """
        if not self.trainer:
            self.setup()

        selected_training : str = self.select_training()
        hyperparameters : dict = self.get_hyperparameters()
        episodes : int = select_value("Episodenanzahl festlegen: ", value_type = "int")

        match selected_training:
            case "base_training":
                self.trainer.base_training(episodes, **hyperparameters)
            case "self_play":
                agent_a : Agent = select_agent()
                agent_b : Agent = select_agent()

                print("Hyperparameter für Agent B:")
                hyperparameters_b: dict = self.get_hyperparameters("biases")

                self.trainer.self_play(episodes, agent_a = agent_a, agent_b = agent_b, **hyperparameters, **hyperparameters_b)
            case "league_play":
                agent_a : Agent = select_agent()
                self.trainer.league_play(episodes, agent_a, None, **hyperparameters)
            case "full_training":
                cycles : int = int(select_value("Anzahl an Trainingswiederholungen (cycles) festlegen: ", value_type = "int"))
                self.trainer.full_training(episodes, cycles, **hyperparameters)

if __name__ == "__main__":
    training = Training()
    training.train()
