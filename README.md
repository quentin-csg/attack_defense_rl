# Attack & Defense RL

Projet d'apprentissage par renforcement adversarial : un **Red Teamer** (attaquant) apprend à compromettre un réseau simulé tandis qu'une **Blue Team** (défenseur) tente de le stopper. Le tout avec Fog of War, génération procédurale de réseaux, et visualisation live.

---

## Présentation

L'idée centrale : faire s'affronter deux agents RL sur un réseau cybersécurité simulé. Le Red Teamer doit exfiltrer des données en restant discret, la Blue Team doit le détecter et l'éjecter. Ni l'un ni l'autre ne connaît la stratégie de l'autre — ils apprennent en jouant.

Ce projet est développé en plusieurs phases, chacune ajoutant une couche de complexité :

1. **Phase 1** — Environnement Gymnasium + Fog of War 
2. **Phase 2** — Visualisation Pygame live
3. **Phase 3** — Entraînement RL du Red Teamer (MaskablePPO)
4. **Phase 4** — Blue Team scriptée réactive
5. **Phase 5** — Génération procédurale de réseaux (PCG)
6. **Phase 6** — Blue Team RL + Self-play adversarial

---

## Stack technique

| Composant | Technologie | Rôle |
| --- | --- | --- |
| Environnement | **Gymnasium** (custom) | Boucle RL standard, contrôle total |
| Graphe réseau | **NetworkX** | Topologie, algorithmes de chemin |
| Entraînement RL | **sb3-contrib** `MaskablePPO` | Action masking intégré |
| Visualisation | **Pygame** | Dashboard live hacker-style |
| Monitoring | **TensorBoard** | Courbes d'entraînement |
| Tests | **pytest** | 266 tests, tous verts |
| Linting | **ruff** | Format + lint |

---

## Architecture

```text
attack_defense_rl/
├── src/
│   ├── config.py                 # Toutes les constantes (rewards, suspicion, probas)
│   │
│   ├── environment/              # Phase 1 
│   │   ├── node.py               # Dataclass Node (OS, services, vulns, session, suspicion)
│   │   ├── vulnerability.py      # VulnType + registre extensible (8 vulns built-in)
│   │   ├── network.py            # Wrapper NetworkX (topologie, isolate/restore)
│   │   ├── fog_of_war.py         # Masque d'observation partielle (Fog of War)
│   │   ├── actions.py            # 14 ActionType + logique d'exécution
│   │   ├── action_mask.py        # Masque booléen pour MaskablePPO
│   │   └── cyber_env.py          # CyberEnv(gymnasium.Env) — env principal
│   │
│   ├── visualization/            # Phase 2 (implémenté — minimal)
│   ├── agents/                   # Phase 3 (implémenté)
│   │   ├── wrappers.py           # ActionMasker + DummyVecEnv factories
│   │   ├── red_trainer.py        # MaskablePPO config + train + evaluate
│   │   └── callbacks.py          # CyberMetricsCallback (TensorBoard)
│   ├── pcg/                      # Phase 5 (à venir)
│   └── utils/
│
├── tests/
│   ├── conftest.py               # Fixtures partagées
│   ├── phase1/                   # 119 tests Phase 1
│   ├── phase2/                   # 43 tests Phase 2
│   └── phase3/                   # 33 tests Phase 3
│
├── scripts/
│   ├── train_red.py              # Lancer l'entraînement RL
│   ├── evaluate.py               # Évaluer un modèle sur N épisodes
│   └── visualize.py              # Visualisation live (agent aléatoire ou entraîné)
└── models/                       # Modèles sauvegardés (gitignored)
```

---

## Fonctionnalités — Phase 1

### Modèle réseau

- Nœuds avec OS type (Linux / Windows / Network Device), services, vulnérabilités, suspicion (0-100), niveau de session (NONE / USER / ROOT)
- Topologie NetworkX — connexions, isolation réversible de nœuds, calcul de chemin
- **Topologie fixe 8 nœuds** pour le développement (DMZ → LAN → Datacenter)

### Système de vulnérabilités extensible

- Types : `RCE`, `SQLI`, `URL_INJECTION`, `PRIVESC`, `BRUTE_FORCE`
- Registre global — ajouter une nouvelle vuln = une seule ligne
- 8 vulnérabilités built-in (rce_generic, sqli_basic, privesc_kernel, weak_credentials, etc.)

### 14 actions Red Teamer

| Action | Effet | Suspicion |
| --- | --- | --- |
| `SCAN` | Découvre les nœuds adjacents | +3 |
| `ENUMERATE` | Révèle services/vulns (discret) | +5 |
| `ENUMERATE_AGGRESSIVE` | Révèle services/vulns (rapide) | +15 |
| `EXPLOIT` | Session USER via RCE/SQLI/URL (probabiliste ~80%) | +10 à +25 |
| `BRUTE_FORCE` | Session USER via credentials faibles | +30 |
| `PRIVESC` | USER → ROOT via vuln privesc | +12 |
| `CREDENTIAL_DUMP` | Extrait des credentials réutilisables | +15 |
| `PIVOT` | Accès à un nœud DISCOVERED non-adjacent via relais compromis | +5 |
| `LATERAL_MOVE` | Accès adjacent via creds dumpés | +8 |
| `INSTALL_BACKDOOR` | Accès persistant (résiste à ROTATE_CREDENTIALS) | +10 |
| `EXFILTRATE` | **Objectif principal** — +150 reward (requiert ROOT) | +20 |
| `TUNNEL` | Chiffrement — divise suspicion future par 2 | +5 |
| `CLEAN_LOGS` | Efface les traces (diminishing returns) | -15/-10/-5/-2 |
| `WAIT` | Réduit la suspicion (floor = max_historique / 2) | -3 |

### Fog of War

- 3 niveaux : `UNKNOWN` / `DISCOVERED` (IP connue) / `ENUMERATED` (services + vulns connus)
- Observation partielle : les nœuds non-découverts ont des features à `-1` (padding sentinel)
- La matrice d'adjacence est masquée pour les nœuds non-découverts

### Action Masking (MaskablePPO)

- Masque booléen de `N_ACTION_TYPES × MAX_NODES = 700` actions
- Invalide automatiquement les actions impossibles (EXPLOIT sur nœud non-énuméré, PRIVESC sans vuln, etc.)
- `WAIT` est **toujours valide** — le masque ne peut pas être tout-à-zéro

### Observation Space (Dict)

```python
{
    "node_features":    Box(-1, 1, shape=(50, 13)),  # features par nœud, -1 = padding
    "adjacency":        Box(0, 1, shape=(50, 50)),   # matrice adjacence masquée
    "node_exists_mask": MultiBinary(50),             # nœuds réels vs padding
    "fog_mask":         MultiBinary(50),             # nœuds découverts vs cachés
    "agent_position":   Discrete(50),               # position actuelle
    "global_features":  Box(0, 1, shape=(3,)),      # step/compromis/découverts
}
```

### Rewards calibrés

```python
REWARD_EXFILTRATE           = +150.0  # ratio = 150 / (0.5 * 200) = 1.5 > 1
REWARD_PER_STEP             =   -0.5  # pression temporelle
REWARD_DETECTED             =  -50.0
REWARD_NEW_NODE_DISCOVERED  =   +2.0
REWARD_NEW_NODE_COMPROMISED =   +5.0
REWARD_ROOT_OBTAINED        =  +10.0
REWARD_REPEATED_ACTION      =   -1.0  # WAIT est exempté
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances principales** :

- `gymnasium>=0.29`
- `networkx>=3.0`
- `sb3-contrib>=2.0` (MaskablePPO)
- `stable-baselines3>=2.0`
- `pygame>=2.5`
- `numpy>=1.24`

---

## Utilisation

### Lancer l'environnement manuellement

```python
from src.environment.cyber_env import CyberEnv
from src.environment.actions import ActionType, encode_action

env = CyberEnv(seed=42)
obs, info = env.reset()

# Récupérer le masque d'actions valides
mask = env.action_masks()

# Exécuter une action (SCAN depuis le nœud 0)
action = encode_action(ActionType.SCAN, 0)
obs, reward, terminated, truncated, info = env.step(action)

print(f"Step: {info['step']}, Reward: {reward:.1f}")
print(f"Nœuds découverts: {info['n_discovered']}")
print(f"Suspicion max: {info['max_suspicion']:.0f}%")
```

### Visualiser un épisode (Phase 2 minimale)

```python
from src.environment.cyber_env import CyberEnv

# Ouvre une fenêtre Pygame et affiche le graphe en temps réel
env = CyberEnv(seed=42, render_mode="human")
obs, info = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()
```

Le graphe affiche :
- **Cyan** : nœud énuméré, non compromis
- **Jaune** : nœud découvert, non énuméré
- **Rouge** : session USER
- **Rouge vif** : session ROOT
- **Gris sombre** : nœud inconnu (Fog of War) ou hors-ligne
- **Anneau vert** : nœud d'entrée (DMZ)
- **Anneau gold** : nœud cible (Data Center)
- **Anneau blanc** : position actuelle de l'agent

Mode headless (pour scripts et tests) :

```python
env = CyberEnv(seed=42, render_mode="rgb_array")
obs, info = env.reset()
frame = env.render()  # np.ndarray (H, W, 3) uint8
env.close()
```

### Entraîner l'agent Red Team (Phase 3)

```bash
# Entraînement complet (500k steps, ~2h CPU)
python scripts/train_red.py

# Entraînement court pour tester
python scripts/train_red.py --timesteps 50000 --eval-freq 5000

# Suivre l'entraînement avec TensorBoard
tensorboard --logdir logs/
```

Métriques TensorBoard disponibles :

- `cyber/exfiltration_rate` — taux d'épisodes réussis (objectif principal)
- `cyber/detection_rate` — taux de détections par la Blue Team
- `cyber/mean_nodes_compromised`, `cyber/mean_max_suspicion`

### Évaluer un modèle entraîné

```bash
# Évaluation sur 100 épisodes
python scripts/evaluate.py models/red_agent_final.zip

# Évaluation avec politique stochastique
python scripts/evaluate.py models/red_agent_final.zip --episodes 200 --stochastic
```

### Visualiser un agent entraîné

```bash
# Visualisation avec agent entraîné (fenêtre Pygame)
python scripts/visualize.py --model models/red_agent_final.zip --speed 3

# Visualisation avec actions aléatoires (baseline)
python scripts/visualize.py --speed 3
```

### Tester l'environnement avec gymnasium

```python
from gymnasium.utils.env_checker import check_env
from src.environment.cyber_env import CyberEnv

env = CyberEnv()
check_env(env)  # doit passer sans warnings
```

### Tests

```bash
# Tous les tests
pytest tests/ -v

# Phase 1 uniquement
pytest tests/phase1/ -v

# Avec coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/ 
```

---

## Progression

| Phase | Description | Statut |
| --- | --- | --- |
| Phase 1 | Environnement Gymnasium + Fog of War |  Terminé |
| Phase 2 min | Visualisation minimale Pygame | Terminé |
| Phase 3 | Entraînement Red (MaskablePPO) | Terminé |
| Phase 2 complète | Visualisation panels + animations | En attente |
| Phase 4 | Blue Team scriptée | En attente |
| Phase 5 | Génération procédurale (PCG) | En attente |
| Phase 6 | Blue Team RL + Self-play | En attente |
