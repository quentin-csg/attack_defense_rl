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

Suivi live Streamlite + TensorBoard :

```bash
streamlit run scripts/dashboard.py
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

## Topologies réseau (PCG)

| Taille | Nœuds | Sous-réseaux | Steps max | Description |
| --- | --- | --- | --- | --- |
| Small | 10 – 15 | 2 – 3 | 150 | Réseau simple, entraînement initial |
| Medium | 25 – 30 | 4 – 5 | 250 | Réseau intermédiaire, difficulté modérée |
| Large | 50 – 60 | 7 – 8 | 400 | Réseau complexe, curriculum final |

Chaque topologie suit une progression de zones : **DMZ → Corporate → Server → Datacenter**. L'objectif Red Team est d'atteindre le nœud cible dans le Datacenter et d'exécuter `LIST_FILES`.

## Actions Red Team

| Action | Condition requise | Suspicion | Effet |
| --- | --- | --- | --- |
| `SCAN` | Nœud découvert ou adjacent | +3 | Révèle l'existence des nœuds voisins |
| `ENUMERATE` | Nœud découvert (exactement) | +5 | Révèle les services et vulnérabilités du nœud |
| `ENUMERATE_AGGRESSIVE` | Nœud découvert (exactement) | +15 | Idem, plus rapide mais très visible |
| `EXPLOIT` | Nœud énuméré + vulnérabilité connue | +10 min | Obtient un accès initial (USER ou ROOT selon la vuln) |
| `BRUTE_FORCE` | Nœud énuméré + `weak_credentials` | +30 | Tente d'obtenir un accès via force brute (90% succès) |
| `PRIVESC` | Accès USER sur le nœud | +12 | Élève les privilèges USER → ROOT |
| `CREDENTIAL_DUMP` | Accès ROOT sur le nœud | +15 | Extrait les credentials pour pivoter vers d'autres nœuds |
| `PIVOT` | Accès ROOT + credentials d'un nœud voisin (2 hops max) | +5 | Se déplace furtivement vers un nœud adjacent déjà compromis |
| `LATERAL_MOVE` | Accès ROOT sur le nœud courant | +8 | Se déplace vers un nœud voisin (sans credentials requis) |
| `INSTALL_BACKDOOR` | Accès ROOT sur le nœud | +10 | Installe une backdoor (survit à `ROTATE_CREDENTIALS`) |
| `TUNNEL` | Accès ROOT sur le nœud | +5 | Réduit de moitié la suspicion générée par les actions suivantes sur ce nœud |
| `CLEAN_LOGS` | Accès sur le nœud | −15 / −10 / −5 / −2 | Réduit la suspicion (rendements décroissants, cooldown 1 step) |
| `WAIT` | Toujours valide | −3 | Laisse passer un step, réduit légèrement la suspicion |
| `LIST_FILES` | Accès ROOT sur le nœud cible (Datacenter) | +5 | **Condition de victoire** — termine l'épisode avec la récompense maximale |
| `EXFILTRATE` | *(legacy)* | +20 | Ancienne condition de victoire, remplacée par `LIST_FILES` |

## Actions Blue Team

| Action | Déclencheur | Effet |
| --- | --- | --- |
| `ALERT` | Suspicion ≥ 60 (±10 bruité) | Marque le nœud sous surveillance — double la suspicion générée par les actions Red suivantes sur ce nœud |
| `ROTATE_CREDENTIALS` | Suspicion ≥ 80 (±10 bruité), cooldown 5 steps | Invalide la session Red sur le nœud (sauf si backdoor installée) |
| `ISOLATE_NODE` | Suspicion ≥ 95 (±5 bruité) | Déconnecte le nœud du réseau — auto-restauré après `max(10, max_steps // 25)` steps |
| `PATROL` | Poisson (intervalle moyen 5 steps, scalé par taille réseau) | Inspecte un nœud : si traces détectables → +25 suspicion + mise sous surveillance. Ciblage adaptatif : nœuds où la suspicion monte récemment sont 5× plus probables, leurs voisins 2.5× |
| `RESTORE_NODE` | Automatique après durée d'isolation | Reconnecte le nœud et lève la surveillance |

> Les seuils ALERT/ROTATE/ISOLATE sont re-randomisés à chaque épisode pour empêcher le Red Team d'apprendre des timings fixes.

## Progression

| Phase | Description | Statut |
| --- | --- | --- |
| Phase 1 | Environnement Gymnasium + Fog of War | Terminé |
| Phase 2 | Visualisation Pygame (panels, animations, replay) | Terminé |
| Phase 3 | Entraînement Red Team (MaskablePPO) | Terminé |
| Phase 4 | Blue Team scriptée réactive | Terminé |
| Phase 5 | Génération procédurale (PCG) + Curriculum | Terminé |
| Phase 6 | Blue Team RL + Self-play adversarial | Optionnel |
