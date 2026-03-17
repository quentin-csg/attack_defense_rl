Phase 1 : Le Terrain de Jeu (Environment Design)
Structure Hybride : Utilisation d'un graphe NetworkX pour la topologie (nœuds et arêtes) traduit dynamiquement en un scénario CybORG.
Attributs de "Fidélité Logique" par nœud :
Services détaillés : Version précise (ex: Apache 2.4.41), ports ouverts, et fichiers sensibles (ex: robots.txt, config.php).
Vulnérabilités spécifiques : Mapping avec des CVE réelles (ex: RCE, Privilege Escalation) au lieu d'un simple flag "vulnérable".
Système de fichiers & Creds : Présence de fichiers de configuration ou de bases de données avec des mots de passe (ex: admin/admin).
Gestion des États : Suivi du suspicion_level (0-100), de l'état is_online, et surtout du niveau de session (None, User, Root).
Le "Fog of War" : Le Red Teamer démarre avec une visibilité nulle et doit construire sa propre carte mentale via des actions de reconnaissance.


Phase 2 : Le Red Teamer (Offensive Agent)
Utilisation des capacités natives de CybORG pour simuler un défenseur actif.
Agents Intégrés : Utilisation d'agents comme B_S02 qui réagissent aux anomalies de réseau.
Mécanismes de Réaction :
Analyse de Log : Détection des scans de ports ou des tentatives de brute-force.
Remédiation : Rotate_Credentials (rend les accès volés obsolètes) ou Isolate_Node (coupe le trafic vers une machine infectée).
Exploit(Target, Vuln) : Tente un hack (80% succès, 5% crash, 15% échec, grosse suspicion).
Privesc
Pivot(Target) : Tente de rebondir vers de nouveaux nœuds depuis une machine déjà compromise.
$$R_{red} = ({Data\_Exfiltrée} \times 1000) - ({Temps\_Passé}) - ({Detection\_Pénalité})$$


Phase 3 : La Blue Team (Defensive Logic)
Au début, fais-en un script simple, puis transforme-le en IA.
Mécanismes de Défense :
Threshold_Check : Si suspicion_level > X, déclencher une alerte.
Rotate_Credentials : Change les MDP d'un nœud (rend les accès de la Red Team obsolètes).
Isolate_Node : Coupe le nœud du réseau (le Red Teamer doit trouver un rerouting)
$$R_{blue} = ({Intrus\_Bloqué} \times 500) - ({Faux\_Positif} \times 100) - ({Downtime\_Système})$$

Phase 4 : Le Générateur Procédural (PCG) & Chaos
C'est ici que tu garantis que l'IA ne "triche" pas en apprenant un réseau fixe.
Topologie Réaliste : Utilisation du modèle Barabási-Albert pour créer des hubs (serveurs centraux, AD) et des feuilles (postes de travail).
Contrainte de K-Connectivité : Algorithme garantissant au moins $k$ chemins entre les zones critiques (DMZ, LAN, Data) pour forcer l'IA à trouver des routes de secours si la Blue Team bloque un chemin.
Distribution Aléatoire des "Failles" : 20% de chances d'un port 80 ouvert avec un fichier robots.txt révélateur.Probabilité de trouver des mots de passe faibles sur les services non critiques (ex: imprimantes, vieux serveurs).
Stochasticité : Le succès d'un exploit n'est jamais garanti à 100% (simulation de l'instabilité des payloads ou des IDS).
Rerouting Obligatoire : Si la Blue Team ferme le chemin principal, le Red Teamer reçoit une énorme 
pénalité s'il s'arrête. Il doit "chercher" de nouvelles connexions (ex: via un vieux serveur d'impression oublié).

Phase 5 : Visualisation & Monitoring
Live Network Map : Un graphe où les nœuds changent de couleur selon leur état (Bleu = Sûr, Jaune = Scanné, Rouge = Compromis, Gris = Crashé/Isolé).
Dashboards (WandB ou TensorBoard) :
Courbe de Mean Reward.
Histogramme des actions : Suivi du ratio Stealth vs. Loud. Tu verras qu'au début l'IA fait 90% de Exploit (bourrin) et qu'à la fin elle fait 70% de Recon (furtif).
Taux de détection par la Blue Team.



Moteur de Simulation : CybORG (pour la gestion des sessions et des actions cyber).
Moteur de Graphe : NetworkX (pour le PCG et la topologie k-connectée).
Entraînement RL : Gymnasium + Stable-Baselines3 (PPO pour sa stabilité dans les environnements complexes).