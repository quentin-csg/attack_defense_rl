# Roadmap -- Attack & Defense RL

> Projet d'apprentissage par renforcement : un Red Teamer (attaquant) apprend à compromettre un réseau
> pendant qu'une Blue Team (défenseur) tente de le stopper. Fog of War, génération procédurale,
> et visualisation live.

---

## Choix techniques globaux

| Composant        | Technologie                  | Justification                                              |
| ---------------- | ---------------------------- | ---------------------------------------------------------- |
| Environnement    | **Gymnasium** (custom)       | Contrôle total, pas de dépendance fragile (exit CybORG)    |
| Graphe / Topologie | **NetworkX**               | Standard, PCG facile, algorithmes intégrés                 |
| Entraînement RL  | **Stable-Baselines3** (PPO)  | Stable, bien documenté, adapté aux espaces complexes       |
| Visualisation    | **Pygame** + **Rich** (terminal) | Rendu live stylé du réseau + dashboard console         |
| Monitoring RL    | **TensorBoard**              | Intégré nativement à SB3, zero config                     |
| Tests            | **pytest**                   | Standard Python                                            |

---

## Partie 1 -- Le Monde : Environnement de base & Fog of War

**Objectif** : Avoir un environnement Gymnasium fonctionnel avec un réseau, des nœuds, et un brouillard de guerre.

### 1.1 -- Modèle de données du réseau
- Classe `Node` : attributs `id`, `os_type`, `services` (liste de dicts), `is_online`, `suspicion_level` (0-100), `session_level` (enum: NONE, USER, ROOT)
- Classe `Network` : wrapper autour d'un graphe NetworkX, méthodes pour ajouter/retirer des nœuds et arêtes
- Topologie initiale fixe (5-8 nœuds) codée en dur pour le développement

### 1.2 -- Système de vulnérabilités extensible
- Classe de base `VulnType` avec un nom (`rce`, `sqli`, `url_injection`, `privesc`, `brute_force`...)
- Registre de vulnérabilités : un simple dict `{nom: VulnType}` facile à étendre
- Chaque `Node` a une liste de `VulnType` actives (pas de versions, juste des types)
- Pour ajouter un nouveau type de vuln plus tard : une seule ligne dans le registre

### 1.3 -- Fog of War (observation partielle)
- L'agent Red ne voit que les nœuds qu'il a **découverts** (via scan/recon)
- Deux couches d'observation :
  - `discovered` : le nœud existe (IP connue)
  - `enumerated` : les services et vulns de ce nœud sont connus
- L'état observable est un masque appliqué sur l'état réel du réseau
- Le reste est rempli de zéros (inconnu)

### 1.4 -- Environnement Gymnasium
- Classe `CyberEnv(gymnasium.Env)` avec :
  - `observation_space` : matrice (nb_max_nodes x nb_features) + masque fog of war
  - `action_space` : `Discrete` ou `MultiDiscrete` (action_type, target_node)
  - `reset()` : recrée le réseau fixe, reset le fog of war
  - `step(action)` : exécute l'action, retourne (obs, reward, done, truncated, info)
- Actions de base du Red Teamer :
  - `SCAN` : découvre les nœuds adjacents (lève le fog) : **+3 suspicion**
  - `ENUMERATE` : révèle les services/vulns d'un nœud découvert, lent mais discret : **+5 suspicion**
  - `ENUMERATE_AGGRESSIVE` : révèle les services/vulns d'un nœud découvert, rapide mais bruyant : **+15 suspicion**
  - `EXPLOIT(target, vuln_type)` : tente un exploit (probabiliste : ~80% succès, ~5% crash, ~15% échec + suspicion) : **+10 à +25 suspicion** (selon le type de vuln, voir barème 1.3)
  - `BRUTE_FORCE(target)` : tente un login par force brute, marche si `weak_credentials` existe : **+30 suspicion**
  - `PRIVESC(target)` : élévation USER -> ROOT (nécessite une vuln de type privesc) : **+12 suspicion**
  - `CREDENTIAL_DUMP(target)` : récupère des credentials réutilisables sur d'autres nœuds : **+15 suspicion**
  - `PIVOT(source, target)` : accès à un nœud non-adjacent via un nœud compromis : **+5 suspicion**
  - `LATERAL_MOVE(source, target)` : utilise des creds dumpés pour accéder à un nœud adjacent : **+8 suspicion**
  - `INSTALL_BACKDOOR(target)` : maintient l'accès même après ROTATE_CREDENTIALS : **+10 suspicion**
  - `EXFILTRATE(target)` : exfiltre des données (objectif principal, donne du reward) : **+20 suspicion**
  - `TUNNEL(source, target)` : crée un tunnel chiffré, les actions via ce tunnel ont ensuite ÷2 suspicion : **+5 suspicion** (création)
  - `CLEAN_LOGS(target)` : efface les traces sur un nœud (nécessite ROOT) : **-15 suspicion** (diminishing returns : -15, -10, -5, -2... à chaque usage consécutif. Cooldown de 1 step : impossible d'en faire 2 d'affilée)
  - `WAIT` : ne rien faire, la suspicion décroît mais le timer tourne : **-3 suspicion/step** (floor : la suspicion ne descend jamais en dessous de max_historique ÷ 2 -- la Blue Team n'oublie pas complètement)
- Reward Red simple : `+data_value` par exfiltration, `-1` par step (pression temporelle), `-50` si détecté

### Fichier de tests : `tests/test_part1.py`
- Test création de nœuds et réseau
- Test ajout/extension de types de vulnérabilités
- Test fog of war (l'agent ne voit pas ce qu'il n'a pas scanné)
- Test env Gymnasium : reset, step, spaces valides
- Test chaque action (scan, enumerate, exploit, pivot, exfiltrate)
- Test que les probabilités d'exploit fonctionnent (succès/échec/crash sur N essais)

---

## Partie 2 -- La Vue : Visualisation live du réseau

**Objectif** : Interface visuelle moderne pour observer le déroulement en temps réel.

### 2.1 -- Rendu du graphe réseau (Pygame + Rich)
- Fenêtre Pygame avec rendu du graphe NetworkX (layout spring ou kamada-kawai)
- Chaque nœud est un cercle avec :
  - **Couleur** selon l'état : Bleu (sûr), Jaune (scanné/énuméré), Rouge (compromis USER), Rouge vif/pulsant (ROOT), Gris (crashé/isolé)
  - **Icône/label** : type de machine (serveur, poste, routeur)
  - **Halo/glow** animé sur le nœud actuellement ciblé par l'agent
- Les arêtes montrent les connexions réseau, avec mise en surbrillance du chemin actif de l'attaquant
- **Fog of War visuel** : les nœuds non découverts sont des ombres floues / silhouettes, les nœuds découverts mais non énumérés sont semi-transparents

### 2.2 -- Panel d'informations
- Panel latéral avec les infos du nœud sélectionné (services, vulns, état)
- Log en temps réel des actions de l'agent (scrolling, avec couleurs par type d'action)
- Barre de suspicion par nœud (gauge colorée)

### 2.3 -- Contrôles de visualisation
- Pause / Play / Step-by-step (pour observer action par action)
- Slider de vitesse de simulation
- Zoom et drag sur le graphe

### 2.4 -- Style et polish
- Thème sombre type "hacker terminal" avec accents néon (vert/cyan/rouge)
- Animations fluides : transitions de couleur, particules lors d'un exploit réussi
- Font monospace stylée (type JetBrains Mono ou similaire)
- Header avec les stats en direct : step count, reward cumulé, nœuds compromis, suspicion max

### Fichier de tests : `tests/test_part2.py`
- Test que le renderer se lance sans crash (mode headless / mock)
- Test mapping état du nœud -> couleur correcte
- Test que le fog of war masque bien les nœuds visuellement
- Test du log d'actions (ajout et affichage)
- Test des contrôles (pause, vitesse)

---

## Partie 3 -- Le Cerveau : Entraînement RL du Red Teamer

**Objectif** : L'agent apprend à compromettre le réseau de manière efficace et furtive.

### 3.1 -- Wrappers & preprocessing
- Wrapper pour aplatir l'observation en vecteur compatible SB3
- Wrapper de normalisation des observations
- Wrapper de logging (enregistre actions, rewards, épisodes pour analyse)

### 3.2 -- Entraînement PPO
- Configuration PPO avec Stable-Baselines3
- Hyperparamètres de départ : lr=3e-4, n_steps=2048, batch_size=64, n_epochs=10
- Entraînement sur le réseau fixe (partie 1) -- objectif : convergence sur un réseau simple
- Callback pour sauvegarder les meilleurs modèles

### 3.3 -- Reward shaping
- Reward de base : exfiltration (+), temps (-), détection (--)
- Bonus de progression : +petit reward pour chaque nouveau nœud découvert/compromis
- Pénalité de répétition : -reward si l'agent refait la même action inutilement
- Tuning itératif basé sur le comportement observé

### 3.4 -- Monitoring TensorBoard
- Courbe de Mean Reward par épisode
- Histogramme de distribution des actions (ratio recon vs exploit vs exfiltrate)
- Taux de détection par épisode
- Métriques custom : nœuds compromis moyen, temps moyen pour premier root

### Fichier de tests : `tests/test_part3.py`
- Test que les wrappers produisent des observations valides
- Test qu'un entraînement court (1000 steps) tourne sans erreur
- Test que le modèle sauvegardé peut être rechargé et predict
- Test que les métriques TensorBoard sont bien loggées
- Test reward shaping : vérifier les valeurs de reward pour des scénarios connus

---

## Partie 4 -- L'Adversaire : Blue Team scriptée

**Objectif** : Un défenseur réactif qui force le Red Teamer à être furtif.

### 4.1 -- Moteur de défense
- La Blue Team agit à chaque step (après le Red)
- Système de règles à seuils (pas de RL pour l'instant) :
  - Si `suspicion_level > 60` sur un nœud -> `ALERT` (log + surveillance renforcée)
  - Si `suspicion_level > 80` -> `ROTATE_CREDENTIALS` (invalide les sessions USER volées sur ce nœud)
  - Si `suspicion_level > 95` -> `ISOLATE_NODE` (coupe le nœud du réseau, le Red perd l'accès)
- La suspicion augmente quand le Red fait des actions bruyantes (exploit raté, scan agressif)
- La suspicion décroît lentement avec le temps (oubli naturel)

#### Seuils bruités (anti-exploitation)
- Les seuils de la Blue Team ne sont **pas connus exactement** par le Red
- À chaque `reset()`, chaque seuil est tiré aléatoirement dans un intervalle : seuil = valeur_base ± 10
  - ALERT : 60 ± 10 (donc entre 50 et 70)
  - ROTATE_CREDENTIALS : 80 ± 10 (entre 70 et 90)
  - ISOLATE_NODE : 95 ± 5 (entre 90 et 100)
- Le Red ne voit que sa propre suspicion, pas les seuils exacts. Il doit apprendre à jouer avec une marge de sécurité
- Ça empêche la stratégie "monter à 59 et WAIT" puisque le seuil ALERT peut être à 50

#### Patrouilles Blue Team (détection indépendante de la suspicion)
- Toutes les **5 steps**, la Blue Team envoie un "bot de patrouille" sur un nœud aléatoire
- Le bot inspecte le nœud et détecte les **traces lourdes** laissées par le Red :
  - `INSTALL_BACKDOOR` -> **détecté** (backdoor = fichier/process suspect)
  - `EXFILTRATE` -> **détecté** (transfert de données anormal dans les logs réseau)
  - `CREDENTIAL_DUMP` -> **détecté** (accès mémoire/fichiers sensibles)
  - `EXPLOIT` ayant crashé le service -> **détecté** (service down = visible)
  - `ENUMERATE` / `SCAN` / `PIVOT` -> **non détecté** (trop discret pour une patrouille)
- Si une trace est détectée : **+25 suspicion immédiat** + le nœud est marqué comme "sous surveillance" (prochaines actions sur ce nœud = suspicion x2)
- `CLEAN_LOGS` efface aussi les traces détectables par les patrouilles -- c'est sa vraie valeur stratégique
- L'agent doit apprendre à **nettoyer ses traces partout**, pas seulement gérer le score de suspicion global
- La patrouille est visible dans la visu : icône de "scan bleu" qui se déplace sur un nœud aléatoire tous les 5 steps

### 4.2 -- Impact sur le Red Teamer
- `ROTATE_CREDENTIALS` : session_level du Red revient à NONE sur le nœud ciblé (sauf si INSTALL_BACKDOOR actif ET non détecté par patrouille)
- `ISOLATE_NODE` : le nœud est coupé du graphe (arêtes supprimées temporairement), le Red doit trouver un autre chemin
- Les actions Blue sont visibles dans la visualisation (flash bleu sur le nœud défendu)

### 4.3 -- Ré-entraînement du Red
- Ré-entraîner le Red Teamer avec la Blue Team active
- Observer l'évolution de la stratégie : l'agent devrait devenir plus furtif
- Comparer les métriques avant/après Blue Team (ratio recon/exploit, taux de détection)

### Fichier de tests : `tests/test_part4.py`
- Test déclenchement des seuils (suspicion -> action Blue)
- Test ROTATE_CREDENTIALS : la session Red est bien invalidée
- Test ISOLATE_NODE : le nœud est bien déconnecté du graphe
- Test décroissance naturelle de la suspicion
- Test que le Red Teamer peut toujours gagner (il existe un chemin viable malgré la défense)
- Test seuils bruités : sur 100 resets, les seuils varient bien dans l'intervalle ± 10
- Test patrouille : toutes les 5 steps, un nœud est inspecté
- Test patrouille détecte INSTALL_BACKDOOR / EXFILTRATE / CREDENTIAL_DUMP mais pas ENUMERATE / SCAN
- Test patrouille + CLEAN_LOGS : les traces sont bien effacées si CLEAN_LOGS a été fait avant la patrouille
- Test que la détection par patrouille ajoute +25 suspicion et marque le nœud "sous surveillance"
- Test diminishing returns de CLEAN_LOGS : -15, -10, -5, -2 sur 4 usages consécutifs
- Test cooldown CLEAN_LOGS : action refusée si faite au step précédent
- Test floor de WAIT : la suspicion ne descend pas en dessous de max_historique ÷ 2

---

## Partie 5 -- Le Chaos : Génération procédurale de réseaux

**Objectif** : Une fois que Red et Blue ont convergé sur un monde, générer un nouveau monde pour forcer la généralisation.

### 5.1 -- Générateur procédural (PCG)
- Topologie Barabási-Albert (hubs naturels + feuilles)
- Paramètres configurables : nb_nœuds (10-50), nb_attachments, seed
- Zones logiques : DMZ, LAN, DATACENTER -- avec des règles de connectivité entre zones
- Vérification empirique de la connectivité (pas de k-connectivité formelle, juste `nx.is_connected()` + relance si non connecté)

### 5.2 -- Distribution aléatoire des propriétés
- Chaque nœud reçoit aléatoirement :
  - Un OS type (linux, windows, network_device)
  - 1 à 3 services (tirage pondéré)
  - 0 à 2 vulnérabilités (parmi le registre extensible de la Partie 1)
  - Probabilité de "loot" (credentials, données sensibles)
- Les nœuds critiques (hubs) ont plus de services mais aussi plus de défenses
- Les nœuds feuilles (imprimantes, vieux serveurs) ont souvent des failles oubliées

### 5.3 -- Calibrage dynamique de la difficulté

**Problème** : Un réseau de 10 nœuds (DC à 3 hops) et un réseau de 40 nœuds (DC à 12 hops) n'ont pas le même budget d'actions. Avec une suspicion fixe, le petit réseau serait trivial et le grand serait injouable.

**Solution** : À chaque génération de monde, le PCG calcule un **score de difficulté** et ajuste les paramètres en conséquence.

#### Score de difficulté (calculé à la génération)
```
min_hops        = plus court chemin DMZ -> Domain Controller (nx.shortest_path)
avg_defense     = moyenne des niveaux de défense des nœuds sur le chemin
vuln_density    = nombre de vulnérabilités exploitables / nombre de nœuds
network_size    = nombre total de nœuds
```

#### Paramètres ajustés dynamiquement

| Paramètre | Formule | Logique réaliste |
|---|---|---|
| `max_steps` | `base_steps (40) + min_hops * 15 + network_size * 2` | Plus le réseau est grand, plus l'agent a de temps |
| `suspicion_decay` | `base_decay (-3) - floor(network_size / 15)` | Plus le réseau est grand, plus il y a du trafic normal = plus le bruit masque l'attaquant = la suspicion décroît plus vite |
| `suspicion_multiplier` | `1.0 + (avg_defense * 0.3)` | Les réseaux bien défendus génèrent plus de suspicion par action |
| Seuils Blue Team | **inchangés** (30/55/75/90/100) | Les seuils restent fixes : c'est la vitesse à laquelle on les atteint qui change |

#### Exemples concrets

| Réseau | min_hops | network_size | max_steps | suspicion_decay | Difficulté ressentie |
|---|---|---|---|---|---|
| Petit facile (8 nœuds, DC à 2 hops) | 2 | 8 | 40 + 30 + 16 = **86** | -3/step | Rapide, peu de marge d'erreur |
| Moyen (20 nœuds, DC à 6 hops) | 6 | 20 | 40 + 90 + 40 = **170** | -4/step | Équilibré |
| Grand complexe (40 nœuds, DC à 10 hops) | 10 | 40 | 40 + 150 + 80 = **270** | -5/step | Long mais le bruit réseau aide |

### 5.4 -- Boucle d'entraînement : convergence puis nouveau monde

**Principe** : on ne génère pas un nouveau réseau à chaque épisode. On **garde le même monde** et on entraîne Red et Blue dessus jusqu'à ce qu'ils convergent (plus d'amélioration). Ensuite seulement, on génère un nouveau monde.

#### Cycle d'entraînement
```
1. GÉNÉRER un nouveau monde (PCG)
2. ENTRAÎNER Red et Blue sur ce monde fixe
   - Red s'entraîne N steps -> fige Red
   - Blue s'entraîne N steps -> fige Blue
   - Boucle jusqu'à convergence
3. DÉTECTER la convergence
4. SAUVEGARDER le modèle
5. Retour à l'étape 1 avec un nouveau monde
```

#### Détection de convergence
- On suit le **mean reward** de Red et Blue sur les M derniers épisodes (fenêtre glissante)
- Convergence = le delta de mean reward entre deux fenêtres consécutives est < seuil (ex: < 1.0) pour **les deux agents**
- Sécurité : un nombre max d'itérations par monde pour éviter les boucles infinies
- Métriques de convergence :
  - `red_reward_delta` : variation du reward moyen Red
  - `blue_reward_delta` : variation du reward moyen Blue
  - `strategy_entropy` : diversité des actions choisies (si trop basse = mode collapse = on force le passage au monde suivant)

#### Pourquoi cette approche
- **Stabilité** : les agents ont le temps de vraiment comprendre un monde avant d'en voir un autre
- **Transfert learning** : les poids appris sur le monde N sont le point de départ du monde N+1, l'agent ne repart pas de zéro
- **Curriculum naturel** : commencer par des petits mondes (faciles à converger), puis augmenter la taille progressivement
- **Observable** : tu peux voir dans la visu comment l'agent maîtrise un monde puis galère sur un nouveau

#### Progression suggérée des mondes
| Monde | Taille | Particularité | Objectif pédagogique pour l'agent |
|---|---|---|---|
| 1 | 5 nœuds | Linéaire, peu de défense | Apprendre les bases (scan -> exploit -> privesc) |
| 2 | 8 nœuds | Avec une branche morte | Apprendre à ne pas perdre de temps |
| 3 | 12 nœuds | Blue Team plus réactive | Apprendre la furtivité (CLEAN_LOGS, TUNNEL) |
| 4 | 20 nœuds | Plusieurs chemins vers le DC | Apprendre à choisir le meilleur chemin |
| 5+ | 20-40 nœuds | Aléatoire PCG | Généralisation |

### 5.5 -- Stochasticité renforcée
- Le taux de succès des exploits varie selon le type de vuln et le nœud
- Événements aléatoires possibles : un service redémarre, un patch est appliqué mid-episode
- Le réseau peut légèrement changer en cours d'épisode (réalisme)

### Fichier de tests : `tests/test_part5.py`
- Test que le PCG génère des réseaux toujours connectés
- Test distribution des vulnérabilités (statistiquement correct sur N générations)
- Test que l'environnement fonctionne avec des réseaux de tailles variées
- Test reproductibilité avec seed fixe
- Test calibrage dynamique : max_steps et suspicion_decay varient correctement avec la taille du réseau
- Test qu'un petit réseau (8 nœuds) a bien moins de steps qu'un grand (40 nœuds)
- Test que la suspicion_decay est plus forte sur un grand réseau
- Test de jouabilité : sur 100 générations aléatoires, un agent parfait (oracle) peut toujours gagner (le calibrage ne crée pas de réseaux impossibles)
- Test que le score de difficulté est bien inclus dans l'observation de l'agent
- Test de convergence : sur un petit monde, les rewards Red et Blue se stabilisent après N itérations
- Test de transfert learning : un agent pré-entraîné sur le monde 1 converge plus vite sur le monde 2 qu'un agent vierge

---

## Partie 6 -- L'Évolution : Blue Team RL (adversarial)

**Objectif** : La Blue Team devient elle aussi un agent RL. Les deux s'entraînent l'un contre l'autre.

### 6.1 -- Environnement Blue
- Observation Blue : état complet du réseau (pas de fog of war pour le défenseur, mais avec du bruit/délai sur les alertes)
- Actions Blue : MONITOR, ALERT, ROTATE_CREDENTIALS, ISOLATE_NODE, RESTORE_NODE
- Reward Blue : `+500` par intrusion bloquée, `-100` par faux positif, `-downtime` par nœud isolé inutilement

### 6.2 -- Entraînement adversarial
- Self-play : Red et Blue s'entraînent en alternance
- Stratégie : entraîner Red 10k steps -> figer Red -> entraîner Blue 10k steps -> figer Blue -> boucle
- Objectif : un équilibre de Nash émergent (ni Red ni Blue ne peut "tricher")

### 6.3 -- Handicap dynamique (anti-1-sided)

**Problème** : si un agent domine l'autre (winrate > 70%), le perdant ne reçoit que du reward négatif et n'apprend plus rien.

**Solution** : un système de handicap automatique qui rééquilibre dynamiquement.

#### Mécanisme
- On suit le **winrate de Red** sur les 100 derniers épisodes (fenêtre glissante)
- Si winrate Red > 70% (Red domine) :
  - `suspicion_decay` réduit de 1 (la suspicion retombe moins vite, Blue a plus de temps pour réagir)
  - Intervalle des patrouilles Blue réduit de 1 step (patrouilles plus fréquentes)
- Si winrate Red < 30% (Blue domine) :
  - `max_steps` augmenté de +10% (Red a plus de temps)
  - `suspicion_decay` augmenté de 1 (la suspicion retombe plus vite)
- Si winrate entre 30% et 70% : aucun handicap, équilibre satisfaisant
- Le handicap est **progressif** : appliqué à chaque évaluation de winrate, pas en un seul coup
- Le handicap a des limites (caps) pour éviter de dénaturer le jeu : suspicion_decay ne descend pas sous -1 et ne monte pas au-dessus de -8, patrouilles pas plus fréquentes que toutes les 2 steps

#### Pourquoi c'est mieux qu'un rééquilibrage manuel
- Automatique : pas besoin de tuner à la main
- Adaptatif : si les agents évoluent et que l'équilibre change, le handicap s'ajuste
- Observable : le handicap actif est affiché dans les métriques TensorBoard et dans la visu

### 6.4 -- Métriques adversariales
- Elo rating pour Red et Blue au fil des générations
- Visualisation de l'arbre stratégique (quelles actions sont choisies selon les situations)
- Détection de "mode collapse" (si un agent apprend une seule stratégie)

### Fichier de tests : `tests/test_part6.py`
- Test environnement Blue isolé (reset, step, spaces)
- Test boucle de self-play (quelques itérations sans crash)
- Test que les rewards Red et Blue sont anti-corrélés
- Test sauvegarde/chargement des deux agents
- Test qu'après N itérations de self-play, les deux agents s'améliorent (reward moyen croissant)
- Test handicap dynamique : si winrate Red > 70%, suspicion_decay diminue et patrouilles plus fréquentes
- Test handicap dynamique : si winrate Red < 30%, max_steps augmente et suspicion_decay augmente
- Test que le handicap respecte les caps (suspicion_decay entre -1 et -8, patrouilles min 2 steps)
- Test que le handicap ne s'applique pas quand winrate entre 30% et 70%

---

## Résumé des priorités

```
Partie 1  ██████████  Fondation (CRITIQUE)
Partie 2  ██████████  Visualisation (CRITIQUE - tu veux voir le bot)
Partie 3  ████████░░  Cerveau RL (le cœur du projet)
Partie 4  ██████░░░░  Blue Team scriptée (crée le défi)
Partie 5  █████░░░░░  PCG (empêche la triche)
Partie 6  ████░░░░░░  Blue Team RL (cerise sur le gâteau)
```

Chaque partie est **jouable et testable indépendamment** dès qu'elle est terminée.
L'idée est qu'à la fin de chaque partie, tu as quelque chose qui **marche et qui se voit**.

---

## Éléments volontairement écartés / simplifiés

| Élément du plan original                | Décision        | Raison                                                         |
| --------------------------------------- | --------------- | -------------------------------------------------------------- |
| CybORG comme moteur de simulation       | **Retiré**      | API instable, dépendance fragile, Gymnasium custom suffit      |
| Versions exactes de services (Apache 2.4.41) | **Simplifié** | Remplacé par des types de vuln extensibles, plus pragmatique |
| K-connectivité formelle                 | **Simplifié**   | Vérification empirique `nx.is_connected()` suffit              |
| Système de fichiers simulé              | **Simplifié**   | Abstrait en "loot items" typés (credentials, data, config)     |
| Agents CybORG intégrés (B_S02 etc.)    | **Retiré**      | Remplacé par Blue Team custom (scriptée puis RL)               |
