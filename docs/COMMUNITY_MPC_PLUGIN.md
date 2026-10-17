# Plugin communautaire A2RL Tube MPC

Ce plugin extrait le controleur et les planificateurs de ce depot. Il fournit
`A2RLTubeMPCControlStrategy`, `A2RLMPCGlobalRacePlanner` et
`A2RLMPCReferenceLocalPlanner`. Il embarque leurs dependances Python internes,
les configurations et les donnees traitees de Yas Marina. Il fonctionne sans
installation de `chrono_a2rl`, Gymnasium, Stable-Baselines3 ou PyChrono.

## Analyse du depot

- `src/avlite_a2rl` contient deja une facade communautaire, mais importe le
  paquet `chrono_a2rl` et depend des ressources du checkout.
- `control/mpc_lateral.py` selectionne le mode lateral. Le profil habituel
  utilise pure pursuit ; ce plugin selectionne explicitement `full_tube`.
- `control/tube_mpc.py` implemente le QP OSQP, le retour auxiliaire et le
  resserrement des contraintes. Le PID produit acceleration et freinage ;
  `SafetySupervisor` borne les commandes finales.
- La reference est la raceline configuree. Le controleur ne suit pas les
  deviations d'un plan local externe : pas de depassement ni d'evitement
  d'adversaires. Associer les planificateurs fournis au controleur.
- Le bridge Chrono et les outils RL restent dans le depot principal. Le plugin
  de commande peut utiliser un bridge AVLite compatible Ackermann.

## Construire depuis le depot source

```powershell
python scripts/build_community_plugin.py
```

Le repertoire `dist/avlite_a2rl_mpc` est la racine du plugin et
`dist/avlite_a2rl_mpc.zip` son archive. L'export refuse d'ecraser un repertoire
existant ; utiliser `--output CHEMIN_NEUF` pour une nouvelle version.
Le code exporte est genere depuis les modules sources : modifier ceux-ci,
puis regenerer. Aucun chemin absolu du poste ni import `chrono_a2rl` ne subsiste.

## Installer dans AVLite

Version cible : AVLite **0.5.3 a 0.5.x**, Python 3.10 ou superieur.
Placer le repertoire `avlite_a2rl_mpc` dans le dossier de plugins AVLite,
ou definir `AVLITE_PLUGINS_DIR` sur son **parent**. Depuis le depot source :

```powershell
python -m pip install -r dist/avlite_a2rl_mpc/requirements.txt
$env:AVLITE_PLUGINS_DIR = (Resolve-Path dist).Path
python -m avlite plugins
```

Enregistrer le plugin dans le profil actif. Dans les parametres du profil,
ajouter ces valeurs aux sections existantes (conserver le bridge et les autres
reglages du simulateur) :

```yaml
c69_apps:
  c62_community_plugins:
    avlite_a2rl_mpc: avlite_a2rl_mpc
  c62_load_plugins: true
c40_execution:
  c40_controller: A2RLTubeMPCControlStrategy
  c40_global_planner: A2RLMPCGlobalRacePlanner
  c40_local_planner: A2RLMPCReferenceLocalPlanner
  c40_control_dt: 0.02
plugins:
  avlite_a2rl_mpc:
    experiment_config: configs/experiments/community_mpc_yas_marina.yaml
```

Le bridge doit utiliser les coordonnees de la piste embarquee, fournir la pose
et la vitesse ego et accepter les commandes Ackermann (braquage en radians,
acceleration en m/s2). Le bridge A2RL du depot exploite aussi les champs
throttle, brake et gear. Pour un autre circuit, fournir un fichier d'experience
absolu et des chemins absolus pour ses ressources externes.

Le profil initial limite la vitesse a 15 m/s pour la validation. La configuration
du vehicule est une approximation de recherche A2RL ; ce plugin ne constitue
pas une validation sur voiture reelle. OSQP est requis pour le MPC ; le code
dispose d'un repli geometrique en cas d'indisponibilite ou d'echec du solveur.
Verifier `controller.lateral.last_solver_status` pour distinguer ces cas.

## Distribution communautaire

Publier le **contenu du repertoire exporte** dans un depot public dedie, puis
proposer son URL au registre `AV-Lab/avlite-community-plugins` avec les categories
`ControlStrategy`, `GlobalPlannerStrategy` et `LocalPlanningStrategy`.
Aucune publication distante n'est effectuee par le script. Avant redistribution,
preciser la licence du code source et conserver l'attribution des donnees TUMFTM
dans `assets/tracks/yas_marina/README.md`.

Contrat et procedure officielle : https://avlite.org/plugin-development/
