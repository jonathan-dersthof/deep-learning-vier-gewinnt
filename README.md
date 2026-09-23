  # Machine Learning - Grenzen und Möglichkeiten von Deep Reinforcement Learning am Beispiel "Vier Gewinnt"

In diesem Branch wurde zu Übungszwecken der gesamte Code des [Hauptprojekts](https://github.com/jonathan-dersthof/deep-learning-vier-gewinnt/tree/main) so umstrukturiert, dass alle PyTorch Funktionalitäten durch eigene NumPy Implementationen ersetzt wurden. Abgesehen vom Adam Optimizer sind alle Klassen vollkommen eigenständig implementiert worden. 
***
  - Technische Eigenschaften
  
Mithilfe von NumPy habe ich ein Deep Q-Network implementiert. Um Instabilität im Training vorzubeugen wurde darüber hinaus ein Target-Network hinzugefügt (daher DDQN). Außerdem verfügt der Agent über einen Replay-Memory-Buffer mit einer gesetzten Maximallänge von 100.000 Zügen. 

Es sind verschiedene Trainingsabläufe implementiert: Das Basistraining, das den Agenten für x Spiele gegen einen reinen Zufallsgegner trainiert, self-play, in welchem ein Agent entweder gegen eine eingefrorene Version seiner selbst spielt oder in welchem zwei beliebige Agenten gegeneinander Spielen und trainieren und zu guter letzt das Herzstück der Facharbeit mit dem league-play, in welchem ein Agent gegen einen Agenten-pool trainiert, was Overfitting verhindern sollte. 

Die Spielumgebung, die Datenspeicherung und Visualisierung sind vollständig Objektorientiert mit Python implementiert. Das Projekt erlaubt es dem Nutzer über ein textbasiertes-Interface Modelle zu trainieren, gegen bestehende zu trainieren oder schlichtweg gegen einen anderen Menschen lokal zu spielen. 

Verwendet wurden Python 3.14 sowie die Bibliotheken NumPy, Pandas und Matplotlib
***
  - Projektstruktur


Das Projekt ist modular aufgebaut und kann aufgeteilt werden in Klassen für den Agenten, die Spielumgebung, das Training und die Datenverabeitung. 

Der Agent setzt sich zusammen aus dem LinearDQN, welches von PyTorch torch.nn.Module erbt und das Netzwerk konkret implementiert und dem Agenten, welcher Methoden zum Lernen, Handeln, etc. enthält. 

Die Spielumgebung ist Primär die VierGewinnt Klasse, in welcher die Logik für Züge und deren Gültigkeit, der Zustand und Methoden zur Überprüfung von Siegesbedinungen enthalten sind. Darüber hinaus dient die Klasse Game als Schnittstelle, mit welcher ein Nutzer mit der Spielumgebung interagieren kann, egal ob gegen einen Agenten oder Menschen. 

Das Training setzt sich aus mehreren Ebenen zusammen. Die unterste Ebene ist Episode. Diese Klasse ist verantwortlich für ein einziges in sich geschlossenes Traingsspiel, in welchem der gesamte Spielablauf stattfindet und die Belohungen für den Agenten verteilt werden. Auf der Ebene darüber ist Session, welche mehrere Episoden verwaltet und unter den gleichen Parametern (abgesehen von den trainierten Agenten) durchführt und mithilfe von der Klasse Logger Trainingsdaten speichert. Die Klase Trainer widerum ist verantwortlich für das Verwalten mehrer Sessions und entsprechend des genauen Traingsablaufs (s.o.). Training ist die oberste Ebene, welche ähnlich zu Game die Schnittstelle für die Menschliche Interaktion darstellt. Das Modul logic stellt allgemeine Hilsfunktionen, primär für die Navigation der Ordnerstruktur, zur Verfügung. 

> Eine Ausführliche [technische Dokumentation](https://github.com/jonathan-dersthof/deep_learning_vier_gewinnt/blob/main/Facharbeit/Dokumentation%20Facharbeit.pdf) für das Hauptprojekt inklusive UML Diagramm und Beschreibung einzelner Methoden ist als PDF dem Repository beigelegt
***
  - Ergebnisse
    
In der Durchführung des Trainings sind einige Grenzen klar geworden, welche einerseits mit dem Ansatz selbst zusammenhängen, aber anderseits auch mit einer fehlerhaften Implementierung meinerseits zusammenhängen. Die gößten verpassten Chancen sind das nicht vollständige Implementieren eines ConvDQN, welche Probleme des Modells mit dem Verständnis von Räumlichkeit lösen würde und das vollständige Verbieten illegaler Züge des Agenten, wodurch der Agent im Training nicht lernt, dass das Spielfeld nach oben hin begrenzt ist. Außerdem ist die Implementierung mit der Klasse Trainer für einen wisssenschaftlichen Versuch zwar gut geeignet, aber wäre für einen Benutzer des Programms über die Textoberfläche zu restriktiv/umständliche.

Für eine ausführlichere Auswertung liegt dem Repository im Main-Branch außerdem die eigentliche [Facharbeit](https://github.com/jonathan-dersthof/deep_learning_vier_gewinnt/blob/main/Facharbeit/Informatik%20Facharbeit%20.pdf) sowie eine Übersicht über die [Trainingsdaten](https://github.com/jonathan-dersthof/deep_learning_vier_gewinnt/blob/main/Facharbeit/Trainingsdaten%20Facharbeit.pdf) bei. 
