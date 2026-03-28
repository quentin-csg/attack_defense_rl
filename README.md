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
| Tests | **pytest** | 410 tests, tous verts |
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
│   │   ├── actions.py            # 15 ActionType + logique d'exécution (dont LIST_FILES)
│   │   ├── action_mask.py        # Masque booléen pour MaskablePPO
│   │   └── cyber_env.py          # CyberEnv(gymnasium.Env) — env principal
│   │
│   ├── visualization/            # Phase 2 (implémenté — complet)
│   ├── agents/                   # Phases 3, 4
│   │   ├── wrappers.py           # ActionMasker + DummyVecEnv factories
│   │   ├── red_trainer.py        # MaskablePPO config + train + evaluate
│   │   ├── blue_scripted.py      # Blue Team scriptée (Phase 4)
│   │   └── callbacks.py          # CyberMetricsCallback (TensorBoard)
│   ├── pcg/                      # Phase 5
│   │   ├── generator.py          # Barabási-Albert + zones (DMZ/CORP/SERVER/DC)
│   │   ├── difficulty.py         # Score de difficulté + is_solvable + max_steps
│   │   └── curriculum.py         # CurriculumManager (Small → Medium → Large)
│   └── utils/
│
├── tests/
│   ├── conftest.py               # Fixtures partagées
│   ├── phase1/                   # 127 tests Phase 1
│   ├── phase2/                   # 101 tests Phase 2
│   ├── phase3/                   # 38 tests Phase 3
│   ├── phase4/                   # 49 tests Phase 4 (+ 21 Blue scripted + 11 intégration + 17 review)
│   └── phase5/                   # 59 tests Phase 5 (generator, difficulty, curriculum, integration)
│
├── scripts/
│   ├── train_red.py              # Lancer l'entraînement RL
│   ├── evaluate.py               # Évaluer un modèle sur N épisodes
│   ├── visualize.py              # Visualisation live (agent aléatoire ou entraîné)
│   └── dashboard.py              # Dashboard Streamlit (monitoring live/replay)
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

### 15 actions Red Teamer

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
| `EXFILTRATE` | Exfiltration sur réseau fixe (requiert ROOT + loot) | +20 |
| `LIST_FILES` | **Objectif PCG** — exécute `ls` sur la cible (requiert USER + loot) | +5 |
| `TUNNEL` | Chiffrement — divise suspicion future par 2 | +5 |
| `CLEAN_LOGS` | Efface les traces (diminishing returns) | -15/-10/-5/-2 |
| `WAIT` | Réduit la suspicion (floor = max_historique / 2) | -3 |

> **Note** : sur les réseaux PCG, la victoire se fait via `LIST_FILES` (session USER suffisante). La cible n'a aucune vulnérabilité — il faut l'atteindre par `LATERAL_MOVE` / `CREDENTIAL_DUMP`.

### Fog of War

- 3 niveaux : `UNKNOWN` / `DISCOVERED` (IP connue) / `ENUMERATED` (services + vulns connus)
- Observation partielle : les nœuds non-découverts ont des features à `-1` (padding sentinel)
- La matrice d'adjacence est masquée pour les nœuds non-découverts

### Action Masking (MaskablePPO)

- Masque booléen de `N_ACTION_TYPES × MAX_NODES = 15 × 64 = 960` actions
- Invalide automatiquement les actions impossibles (EXPLOIT sur nœud non-énuméré, PRIVESC sans vuln, etc.)
- `WAIT` est **toujours valide** — le masque ne peut pas être tout-à-zéro

### Observation Space (Dict)

```python
{
    "node_features":    Box(-1, 1, shape=(64, 13)),  # features par nœud, -1 = padding
    "adjacency":        Box(0, 1, shape=(64, 64)),   # matrice adjacence masquée
    "node_exists_mask": MultiBinary(64),             # nœuds réels vs padding
    "fog_mask":         MultiBinary(64),             # nœuds découverts vs cachés
    "agent_position":   Discrete(64),               # position actuelle
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

## Fonctionnalités — Phase 2 (visualisation)

### Dashboard Pygame

- **Graphe central** — icônes géométriques par OS type (monitor Windows, rack Linux, shield NETWORK_DEVICE)
- **Fog of War** — nœuds inconnus affichés en nuage gris ; nœuds découverts en cyan/jaune/rouge selon session
- **Panneaux latéraux repliables** — Scan / Nodes / Attacker / Stats / Haze (sidebar gauche)
- **Action Log scrollable** — horodaté, coloré par type (vert=Red, cyan=Blue, rouge=échec)
- **Barres de suspicion** — histogramme bas-gauche, couleur par seuil (vert/jaune/orange/rouge)
- **Animations** — pulse ROOT, flash exploit, halo multi-ring sur la position agent, chemin attaquant en bleu

### Contrôles clavier / souris

| Touche / Action | Effet |
| --- | --- |
| `ESPACE` | Pause / reprendre |
| `+` / `-` | Vitesse × 2 / ÷ 2 |
| `←` / `→` | Replay pas à pas (historique de l'épisode) |
| `←` / `→` maintenu | Accélération progressive (1 → 3 → 6 steps/frame) |
| `G` | Barre de recherche — sauter directement à un step |
| `R` | Redémarrer l'épisode (nouveau réseau en mode PCG) |
| `Z` | Réinitialiser zoom et pan |
| `ESC` | Quitter |
| Molette souris (graphe) | Zoom centré sur le curseur (0.2× à 4.0×) |
| Clic sur un nœud | Afficher le panneau d'info (OS, services, vulns, session, suspicion) |
| Clic + glisser un nœud | Repositionner le nœud librement |
| Clic + glisser le fond | Déplacer (pan) tout le graphe |

### Replay historique

- Chaque step réel est enregistré avec un snapshot complet (`copy.deepcopy`)
- En replay : les nœuds non encore découverts à ce step s'affichent en icône grise (ni nuage, ni rouge final)
- La barre de statut indique `REPLAY step X [Y/Z]`

### Indicateurs Blue Team visuels

- **Bouclier bleu** en haut à droite d'un nœud = nœud actuellement sous surveillance Blue Team
- **BLUE ALERT** logué une seule fois par nœud (pas de spam à chaque step) ; "surveillance lifted" logué à la restauration

---

## Fonctionnalités — Phase 4

### Blue Team scriptée (défenseur réactif)

4 actions disponibles :

| Action | Effet | Déclencheur |
| --- | --- | --- |
| `ALERT` | Marque le nœud en surveillance (suspicion ×2) | suspicion ≥ 60 ± 10 |
| `ROTATE_CREDENTIALS` | Invalide la session Red sur le nœud | suspicion ≥ 80 ± 10 |
| `ISOLATE_NODE` | Déconnecte le nœud du réseau (auto-restore) | suspicion ≥ 95 ± 5 |
| `PATROL` | Détection stochastique des traces (+25 susp) | Processus de Poisson (1/5 steps) |

Caractéristiques :

- **Seuils bruités** — re-randomisés à chaque épisode (±10/±10/±5) pour que le Red ne puisse pas timer les déclenchements
- **Patrouilles Poisson** — timing imprévisible, même fréquence moyenne (CORRECTION 3)
- **Isolation temporaire** — les nœuds isolés sont auto-restaurés après 10 steps (`BLUE_ISOLATE_DURATION`)
- **Cooldown ROTATE** — 5 steps minimum entre deux rotations sur le même nœud (`BLUE_ROTATE_COOLDOWN`)
- **CLEAN_LOGS** — peut descendre sous le floor `max_historical/2` (bypass activé)
- **ROTATE invalide les creds** — après ROTATE_CREDENTIALS, `has_dumped_creds` est remis à False

---

## Fonctionnalités — Phase 5

### Génération procédurale de réseaux (PCG)

3 tailles de réseau avec topologie zonée réaliste :

| Taille | Nœuds | Sous-réseaux | max_steps |
| --- | --- | --- | --- |
| **Small** | 10-15 | 2-3 | 150 |
| **Medium** | 25-30 | 4-5 | 250 |
| **Large** | 50-60 | 7-8 | 350 |

Architecture zonée (Barabási-Albert intra-zone, m=1 — graphe sparse) :

- **DMZ** — point d'entrée (Linux, Log4Shell, Shellshock)
- **CORPORATE** — postes de travail (Windows 60%, EternalBlue, PrintNightmare, weak_creds)
- **SERVER** — serveurs internes (Linux 70%, Docker Escape, SQLI)
- **DATACENTER** — cible finale (Linux 80%, **aucune vuln** — flag.txt uniquement, victoire via `LIST_FILES`)

Connexions inter-zones séquentielles (~25% de chance d'une seconde gateway). Pas de shortcuts cross-zone.

**Cible** : nœud DATACENTER le plus éloigné de l'entrée (distance BFS maximale) — garantit un chemin long.

### Score de difficulté

```python
score = min_hops * 3.0 + n_nodes * 0.5 - path_vuln_density * 5.0 - path_weak_creds * 3.0
```

`is_solvable()` vérifie : chemin entry→target, `has_loot` sur la cible (ROOT non requis — USER suffit pour `LIST_FILES`). Retry ×10 automatique, fallback garanti si échec.

### Curriculum learning

```text
Stage 1 (Small)  → 5 mondes × 100k timesteps
Stage 2 (Medium) → 5 mondes × 150k timesteps
Stage 3 (Large)  → 5 mondes × 200k timesteps
```

### Entraînement PCG

```bash
# Réseau aléatoire petit
python scripts/train_red.py --pcg small --timesteps 100000 --run-name pcg_small

# Curriculum automatique Small → Medium → Large
python scripts/train_red.py --pcg curriculum --run-name curriculum

# Avec Blue Team active
python scripts/train_red.py --pcg medium --blue-team --timesteps 200000
```

### Utilisation de la factory PCG

```python
from src.pcg.generator import NetworkSize, generate_network
from src.environment.cyber_env import CyberEnv
from src.agents.wrappers import make_pcg_masked_env

# CyberEnv avec génération d'un nouveau réseau à chaque reset()
def factory(seed):
    net, _ = generate_network(NetworkSize.SMALL, seed=seed)
    return net

env = CyberEnv(network_factory=factory, max_steps=150, seed=42)
obs, info = env.reset()  # nouveau réseau Small aléatoire

# Ou directement via le wrapper MaskablePPO
env = make_pcg_masked_env(size="medium", seed=42)
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

### 1. Vérifier l'installation (tests)

```bash
# Tous les tests (410, ~4 min)
pytest tests/ -q

# Phase spécifique uniquement
pytest tests/phase5/ -v

# Avec coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### 2. Visualiser un épisode (fenêtre Pygame)

```bash
# Réseau fixe, actions aléatoires
python scripts/visualize.py --speed 3

# Réseau PCG aléatoire — nouveau réseau à chaque épisode
python scripts/visualize.py --pcg small --speed 3
python scripts/visualize.py --pcg medium --speed 2
python scripts/visualize.py --pcg large --speed 2

# Avec un agent entraîné
python scripts/visualize.py --model models/mon_agent_final.zip --speed 5
python scripts/visualize.py --pcg small --model models/pcg_small_final.zip --speed 5
```

Contrôles : `ESPACE` pause/reprendre · `+`/`-` vitesse · `←`/`→` replay · `G` jump · `Z` reset zoom · molette zoom · `R` restart · `ESC` quitter

```bash
# Avec Blue Team active et seed fixe (déterministe)
python scripts/visualize.py --pcg medium --blue-team --speed 3 --seed 42
```

Couleurs des nœuds :

- **Cyan** : énuméré, non compromis
- **Jaune** : découvert, non énuméré
- **Rouge** : session USER · **Rouge vif** : session ROOT
- **Gris sombre** : inconnu (Fog of War) ou isolé par Blue Team
- **Anneau vert** : nœud d'entrée (DMZ) · **Anneau gold** : cible
- **Bouclier bleu** (coin haut-droit) : nœud sous surveillance Blue Team

### 3. Entraîner l'agent Red Team

```bash
# Réseau fixe — bon point de départ (~5 min CPU)
python scripts/train_red.py --timesteps 100000 --run-name baseline

# Réseau fixe + Blue Team active
python scripts/train_red.py --timesteps 300000 --run-name vs_blue --blue-team

# Réseau PCG aléatoire (nouveau réseau à chaque épisode)
python scripts/train_red.py --pcg small --timesteps 100000 --run-name pcg_small
python scripts/train_red.py --pcg medium --timesteps 200000 --run-name pcg_medium
python scripts/train_red.py --pcg large --timesteps 300000 --run-name pcg_large

# PCG + Blue Team
python scripts/train_red.py --pcg medium --blue-team --timesteps 200000 --run-name pcg_medium_blue

# Curriculum automatique Small → Medium → Large (~750k steps total, ~1h CPU)
python scripts/train_red.py --pcg curriculum --run-name curriculum
```

Suivre l'entraînement en live :

```bash
tensorboard --logdir logs/
# Ouvrir http://localhost:6006
```

Métriques clés :

- `cyber/exfiltration_rate` — taux de réussite (objectif : >50%)
- `cyber/detection_rate` — taux de détections Blue Team
- `cyber/mean_nodes_compromised`, `cyber/mean_max_suspicion`

### 4. Évaluer un modèle entraîné

```bash
# Évaluation sur 100 épisodes (déterministe)
python scripts/evaluate.py models/mon_agent_final.zip

# Plus d'épisodes, politique stochastique
python scripts/evaluate.py models/mon_agent_final.zip --episodes 200 --stochastic
```

### 5. Dashboard Streamlit (monitoring d'entraînement)

```bash
streamlit run scripts/dashboard.py
# Ouvrir http://localhost:8501
```

- Mode **Live** : auto-refresh pendant un entraînement en cours
- Mode **Replay** : scrubbing par timestep sur un run terminé
- Courbes TensorBoard, métriques cyber, graphe réseau, checkpoints

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

---

## Progression

| Phase | Description | Statut |
| --- | --- | --- |
| Phase 1 | Environnement Gymnasium + Fog of War | Terminé |
| Phase 2 min | Visualisation minimale Pygame | Terminé |
| Phase 3 | Entraînement Red (MaskablePPO) | Terminé |
| Phase 2 complète | Visualisation panels + animations | Terminé |
| Review pré-Phase 4 | Corrections bugs + vulns réalistes + dashboard Streamlit | Terminé |
| Phase 4 | Blue Team scriptée | Terminé |
| Review pré-Phase 5 | 9 corrections + 17 tests | Terminé |
| Phase 5 | Génération procédurale (PCG) | Terminé |
| Phase 6 | Blue Team RL + Self-play | En attente |
