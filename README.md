# Attack & Defense RL

RL adversarial : un **Red Teamer** apprend à compromettre un réseau simulé, une **Blue Team** tente de le stopper. Fog of War, génération procédurale de réseaux (PCG), visualisation Pygame live.

## Stack

| Composant | Technologie |
| --- | --- |
| Environnement | Gymnasium custom |
| Graphe réseau | NetworkX |
| Entraînement RL | sb3-contrib `MaskablePPO` |
| Visualisation | Pygame |
| Monitoring | TensorBoard |
| Tests | pytest (410 tests) |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Visualiser un épisode

```bash
# Réseau fixe, actions aléatoires
python scripts/visualize.py --speed 3

# Réseau PCG
python scripts/visualize.py --pcg medium --blue-team --speed 3

# Avec un agent entraîné
python scripts/visualize.py --model models/mon_agent.zip --pcg small --speed 5
```

Contrôles : `ESPACE` pause · `+`/`-` vitesse · `←`/`→` replay · `G` jump · molette zoom · `R` restart · `ESC` quitter

### Entraîner l'agent Red Team

```bash
# Réseau fixe (point de départ, ~5 min CPU)
python scripts/train_red.py --timesteps 100000 --run-name baseline

# PCG + Blue Team
python scripts/train_red.py --pcg medium --blue-team --timesteps 200000 --run-name pcg_medium_blue

# Curriculum complet Small → Medium → Large (~11M steps)
python scripts/train_red.py --pcg curriculum --blue-team --seed 45 --run-name curriculum_blue
```

Suivi live :

```bash
tensorboard --logdir logs/
```

Métriques clés : `cyber/exfiltration_rate`, `cyber/detection_rate`, `cyber/mean_nodes_compromised`

### Évaluer un modèle

```bash
python scripts/evaluate.py models/mon_agent.zip --episodes 100
```

### Tests

```bash
pytest tests/ -q
```

## Progression

| Phase | Description | Statut |
| --- | --- | --- |
| Phase 1 | Environnement Gymnasium + Fog of War | Terminé |
| Phase 2 | Visualisation Pygame (panels, animations, replay) | Terminé |
| Phase 3 | Entraînement Red Team (MaskablePPO) | Terminé |
| Phase 4 | Blue Team scriptée réactive | Terminé |
| Phase 5 | Génération procédurale (PCG) + Curriculum | Terminé |
| Phase 6 | Blue Team RL + Self-play adversarial | À venir |
