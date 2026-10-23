# Architecture de l'assistant Bedrock

Document de référence synchronisé avec les salves de refonte de septembre 2026.

## Arborescence principale

```text
app_chat/
├── app.py
├── llm_engine.py
├── ARCHITECTURE.md
├── core/
│   ├── models.py
│   ├── prompt_rewriter.py
│   ├── memory_tools.py
│   ├── conversation_store.py
│   └── multi_agent/
│       ├── ledger.py
│       ├── workers.py
│       ├── roles.py
│       ├── planning.py
│       ├── orchestrator.py
│       ├── background.py
│       ├── run_store.py
│       └── ui.py
├── tools/
│   ├── custom_tools.py
│   └── write_tools.py
├── workspace/
├── proposals/
├── backups/
└── conversations/
```

## Racines et sécurité

- `PROJECT_ROOT` est détecté dans `tools/custom_tools.py`.
- `WORKSPACE_ROOT` vaut par défaut `PROJECT_ROOT/workspace` et peut être
  surchargé par `APP_WORKSPACE_PATH`.
- Tous les chemins transmis aux outils sont relatifs et confinés dans
  `WORKSPACE_ROOT`.
- `informations_workspace()` vérifie réellement création, lecture et
  suppression d'un fichier temporaire.
- Les outils de consultation sont non destructifs. Cela ne signifie pas que le
  système de fichiers est monté en lecture seule.
- Le modèle ne reçoit aucun outil d'approbation ou d'écriture directe.

## Mode chat classique

### Flux d'un tour

1. `st.chat_input` reçoit texte et pièces jointes.
2. La reformulation est exécutée uniquement si l'utilisateur active le toggle.
3. `build_langchain_messages()` injecte la mémoire durable et applique la
   fenêtre glissante.
4. `agent.invoke()` reçoit `recursion_limit=60`.
5. L'application collecte texte, tokens, appels et résultats d'outils.
6. Les refus de proposition sont affichés explicitement.
7. Messages et mémoire sont persistés atomiquement.

### Reformulation

`core/prompt_rewriter.py` doit préserver actions, fichiers, valeurs,
contraintes, formats et demandes d'outils. Les demandes détaillées restent
reformulées tant que le toggle est actif. La borne technique actuelle est de
20 000 caractères et 4 096 tokens de sortie.

### Mémoire et contexte

- Fenêtre glissante : 10 messages bruts récents.
- La fenêtre ne commence pas volontairement par un message assistant lorsque
  des messages utilisateur récents sont disponibles.
- Mémoire durable : résumé, tâches, décisions, fichiers centraux et questions
  ouvertes.
- L'outil mémoire retourne `changed=false` si la mise à jour est identique et
  demande alors au modèle de terminer sa réponse.
- Le bloc mémoire est explicitement décrit comme non autoritatif.

### Persistance des conversations

- Un JSON par conversation sous `PROJECT_ROOT/conversations`.
- Schéma versionné (`schema_version=2`).
- Messages persistés : rôle, contenu, libellé modèle et métadonnées légères des
  pièces jointes, jamais les octets.
- Mémoire normalisée avant écriture et après lecture.
- Écriture atomique avec fichier temporaire unique, `flush`, `fsync` puis
  `os.replace`.
- Maximum 30 conversations et 200 messages par conversation.

## Outils de workspace

### Consultation

`tools/custom_tools.py` expose :

- `informations_workspace`
- `lister_fichiers`
- `explorer_workspace_cible`
- `informations_fichier`
- `lire_fichier`
- `rechercher_dans_fichiers`

L'exploration ciblée classe les fichiers candidats avant lecture détaillée.
L'objectif est de lire les zones nécessaires puis de produire le livrable,
pas d'effectuer une exploration exhaustive.

### Propositions

`tools/write_tools.py` expose uniquement :

- `proposer_modification_fichier`
- `proposer_remplacement_cible`
- `proposer_insertion_ciblee`
- `proposer_suppression_ciblee`

Les propositions sont persistées sous
`PROJECT_ROOT/proposals/proposals.json`. Une proposition identique déjà en
attente est réutilisée au lieu d'être dupliquée.

Une proposition dont le contenu final est syntaxiquement invalide n'est
**jamais créée** : `_register_proposal()` la refuse à la source et renvoie à
l'agent la cause la plus probable (contenu coupé par la limite de tokens) et la
marche à suivre (patch ciblé plutôt que réécriture complète). Auparavant, une
telle proposition apparaissait dans l'interface sans pouvoir être approuvée.

### Approbation

Pour le chat classique, seule l'interface peut appeler `approve_proposal()` ou
`reject_proposal()`. Les workers multi-agents utilisent des outils d'écriture
directe distincts : ceux-ci appliquent la modification avec sauvegarde et
validation syntaxique, sans proposition en attente.

L'interface propose aussi une validation groupée (`approve_pending_proposals`)
et un mode optionnel de validation automatique. La validation groupée constitue
un change set transactionnel : toutes les propositions sont prévalidées sur un
contenu virtuel, puis écrites ensemble. Si une prévalidation ou une écriture
échoue, aucune proposition du groupe n'est déclarée appliquée et les fichiers
déjà remplacés sont restaurés. L'affichage est rafraîchi par un fragment
Streamlit périodique afin de faire apparaître les nouvelles propositions sans
actualisation manuelle.

Les agents peuvent vérifier leurs changements avec `executer_commande_controlee`.
L'outil n'utilise jamais de shell, limite l'exécutable à des outils de test ou
de qualité autorisés, borne la durée et tronque les sorties.

### Objectifs de consommation non bloquants

Les workers reçoivent une cible indicative de tokens de sortie. Cette cible
sert à encourager un résultat factuel et sans répétition, mais ne déclenche
jamais un arrêt, un échec ou un blocage. Le worker doit aller au bout de sa
tâche lorsque davantage de contexte est nécessaire. Un dépassement est
simplement enregistré dans `TaskMetrics.target_output_exceeded` pour permettre
le suivi et l'amélioration des prompts.
À l'approbation :

1. vérification du statut et de la syntaxe ;
2. comparaison de l'empreinte du fichier ;
3. réapplication ciblée si le fichier a changé ;
4. validation syntaxique du contenu final ;
5. création d'une sauvegarde ;
6. écriture atomique ;
7. persistance du statut final.

## Mode multi-agent

### Protocole de coordination compact

Chaque worker retourne un objet JSON court contenant l'état de l'artefact,
son résumé, ses références, ses preuves, ses tests et ses affirmations
structurées. En mode multi-agent, les modifications sont écrites directement
dans le workspace par des outils bornés, sauvegardés et validés; le circuit
d'approbation reste réservé au chat classique. Le `TaskLedger` transforme cette sortie en `Artifact`
persistant. Une sortie historique non structurée reste acceptée, mais elle est
marquée comme telle et n'alimente pas automatiquement le registre de preuves.

Les états d'artefact sont :

`observed -> proposed -> approved -> applied -> tested -> verified`

Une tâche peut demander `completed`, `applied`, `tested` ou `verified` via
`dependency_state`. Elle ne devient exécutable que lorsque ses dépendances ont
atteint cet état. Après approbation dans l'interface, le ledger réconcilie les
identifiants de propositions avec leur état persistant et actualise la révision
du workspace.

`coordination.compile_context()` remplace la concaténation des rapports. Il
transmet uniquement une découverte tronquée, les résumés d'artefacts directement
dépendants et les contradictions ouvertes concernant la tâche. Les objectifs
de taille restent souples et ne bloquent jamais l'exécution.

Une révision stable du workspace est enregistrée avant et après chaque tâche.
Les preuves de fichiers portent également un hash de contenu lorsqu'il peut
être calculé. Deux affirmations structurées incompatibles sur une même révision
créent une contradiction dans le ledger. La revue croisée reçoit le delta
structuré et ses preuves, jamais le rapport narratif complet.

Le tableau de métriques affiche les tokens, le taux de répétition, les fichiers
réellement déclarés comme lus, les artefacts validés et les tokens par artefact
validé. Ces mesures sont informatives et ne déclenchent aucun arrêt.

### Architecture

- Orchestrateur hiérarchique central.
- Chef de projet au tier `expert`.
- Workers assignables aux tiers `nano`, `fast`, `smart`.
- Communication en étoile via `TaskLedger`.
- Exécution parallèle avec `ThreadPoolExecutor`, jamais `asyncio.gather`.
- Contexte d'un worker limité aux livrables de ses dépendances.

### Interface et appels en cours

Streamlit relance le script à chaque interaction. Une opération longue exécutée
dans le flux du script est donc interrompue dès qu'un widget est touché, et
reste muette jusqu'à son terme.

Les opérations longues sont désormais **exécutées dans un thread dédié**
(`core/multi_agent/background.py`) et le script se contente d'observer.

- `BackgroundJob` porte l'état (`idle` → `running` → `done` / `error` /
  `cancelled`), une file d'événements, le résultat, l'erreur et sa trace.
- `ChefDeProjet.set_progress_callback()` relaie chaque `log()` vers la file :
  l'avancement du Chef est visible sans attendre la fin.
- `workers.run_tasks_parallel(..., progress=...)` publie l'attente de créneau,
  l'appel modèle, le mode retenu (lecture / édition / rédaction), les reprises
  et l'avancement de la vague.
- Un observateur défaillant ne peut jamais interrompre un worker ni le Chef :
  les exceptions du callback sont absorbées.

Opérations déportées : interview, découverte, planification, préparation
complète du chantier, exécution, relance ciblée, revue croisée, synthèse et
re-planification.

Pendant un job, `render_multi_agent()` n'affiche que le moniteur : aucun widget
de phase n'est exposé, donc aucun clic ne peut interrompre le travail. Le
moniteur est un `st.fragment(run_every=1s)` qui affiche l'état des tâches en
direct et le journal des événements. À la fin, le fragment déclenche un
`st.rerun(scope="app")` et `_finalize_job()` applique le résultat — toutes les
écritures dans `st.session_state` ont lieu dans le thread du script, jamais
dans le thread de travail.

**Interruption** : le bouton « Interrompre l'opération » arme un
`threading.Event` coopératif. Un appel Bedrock déjà parti va jusqu'à son terme,
mais aucune vague supplémentaire n'est lancée. Les livrables déjà produits sont
conservés : `execute()` sauvegarde après chaque vague.

**Sauvegarde depuis un thread** : `_save_current_run` lit `st.session_state`,
qui n'existe pas hors du script. Le job installe donc un callback qui fige
l'état du toggle au démarrage et n'appelle plus que `save_run` (pur I/O).

L'équipe mobilisée est affichée dans un tableau unique (agent, état, tâche,
tier, modèle réel, dépendances, verdict de revue, tokens) au lieu de colonnes
par tier devenues illisibles au-delà de trois tiers.

### Hiérarchie de supervision

La hiérarchie actuelle est volontairement **à un seul niveau de délégation** :
un Chef de projet planifie et supervise un DAG de workers. Les tiers
`nano → fast → smart → expert` représentent des capacités et des budgets de
modèle ; ils ne représentent pas des managers récursifs.

Ce choix suit un pattern de production plus robuste pour ce type d'application :

- le plan est une structure déclarative compacte, validée avant exécution ;
- les dépendances forment un DAG borné, exécuté par vagues parallèles ;
- les erreurs transitoires peuvent être relancées au même tier, sans escalade
  coûteuse sur les erreurs permanentes ;
- une revue indépendante est réservée aux livrables critiques ;
- les workers peuvent écrire directement dans le workspace; l'humain garde la
  visibilité sur le plan et la synthèse, sans approbation fichier par fichier ;
- les coûts, les appels et les erreurs restent observables dans le ledger.

Un sous-chef ou une délégation récursive ne doit être ajouté que si un besoin
mesurable le justifie (décomposition dynamique, budget séparé, critères
d'acceptation propres et profondeur maximale). Sans ces garde-fous, la
récursion augmente les coûts, la latence et le risque de boucles sans apporter
de garantie supplémentaire.

### Capacités et budgets de plan

Chaque tâche peut maintenant déclarer ses capacités (`read`, `propose_write` ou
`generate`, avec `auto` pour conserver la détection de compatibilité). Le worker
privilégie cette déclaration et ne retombe sur les heuristiques textuelles que
pour les anciens chantiers.

Le plan borne également `max_retries` (0 à 2) et `max_escalations` (0 à 1) par
tâche. Le ledger conserve ces paramètres pour les reprises et les restaurations.
Avant l'exécution, l'interface affiche une estimation conservative en unités
relatives de coût, calculée à partir du tier, des retries, des escalades et des
revues. Il s'agit d'un indicateur de budget, pas d'un prix AWS.

### Découverte déterministe du workspace (DISCOVERY)

Le Chef est instancié avec `create_chat_model()` : c'est un **modèle simple
sans outil LLM**, contrairement à l'agent LangGraph du mode chat. Ce choix est
volontaire. La découverte n'est pas déléguée à un agent qui déciderait quels
outils appeler ; elle est **entièrement déterministe côté Python**, ce qui la
rend prévisible, bornée et testable sans appel Bedrock.

Principes :

- `llm_engine.WORKSPACE_READER_TOOLS` expose uniquement les outils de
  consultation (`informations_workspace`, `lister_fichiers`,
  `explorer_workspace_cible`, `informations_fichier`, `lire_fichier`,
  `rechercher_dans_fichiers`). Aucun outil de proposition, d'approbation,
  d'écriture directe ni de mémoire.
- `llm_engine.get_workspace_reader_agent()` construit un agent React limité à
  ces outils. Il est conservé pour des usages hors mode multi-agent et pour les
  tests ; **le Chef ne l'utilise pas**.

Pendant DISCOVERY, `ChefDeProjet.discover(terms=...)` appelle directement les
outils Python de consultation et construit un `DiscoveryReport` :

- appelle réellement `informations_workspace` (`workspace_info`) ;
- si des termes sont fournis, appelle `explorer_workspace_cible` puis extrait
  les chemins pertinents (`explored_files`) ;
- lit uniquement les fichiers pertinents avec `lire_fichier`
  (`file_contents`), en évitant les lectures redondantes ;
- effectue des recherches ciblées avec `rechercher_dans_fichiers`
  (`search_results`) ;
- compte les appels réels d'outils (`tool_calls`) ;
- borne l'exploration par `MAX_DISCOVERY_FILES` (8 fichiers),
  `MAX_DISCOVERY_SEARCHES` (5 recherches) et `DISCOVERY_FILE_READ_CHARS`
  (6000 caractères par fichier) ;
- produit un contexte compact via `DiscoveryReport.to_context()`, borné à
  `DISCOVERY_CONTEXT_MAX_CHARS`, injecté dans la planification (`chief.plan`).

Ce contexte réel du workspace précède toujours la planification, sans jamais
demander à l'utilisateur de confirmer que les outils sont actifs ni de coller
des fichiers accessibles.

### États du chantier

```text
DISCOVERY ciblée -> INTERVIEW si nécessaire -> PLANIF -> validation du plan -> EXEC
EXEC -> REVIEW sélective -> SYNTHESE
Tout état peut évoluer vers CANCELLED selon le flux d'interface.
```

La découverte déterministe des consignes et fichiers du projet précède
l'entretien. Le chef ne pose une question que si la réponse n'est pas
déductible du workspace et change réellement le plan ou un critère.

Les états connus sont :

- `INTERVIEW`
- `DISCOVERY`
- `PLANIF`
- `EXEC`
- `REVIEW`
- `SYNTHESE`
- `CANCELLED`

Un état inconnu produit une erreur visible.

### Statuts des tâches

- `à faire`
- `en cours`
- `terminé`
- `échec`
- `bloqué`
- `à revoir`
- `annulé`

Le ledger est la source de vérité. Les workers retournent un `WorkerResult` et
`apply_results()` appelle les méthodes métier du ledger.

### Planification

Le Chef produit un JSON contenant : objectif, hypothèses, tâches, tier,
dépendances, critères d'acceptation, forme de sortie et besoin de revue.

`planning.py` :

- limite le plan à 24 tâches ;
- génère les identifiants côté Python ;
- exige au moins un critère d'acceptation ;
- limite les dépendances à des tâches antérieures ;
- détecte auto-dépendances, dépendances inconnues et cycles ;
- conserve les hypothèses dans les décisions du ledger.

#### Tolérance aux défauts JSON

Les modèles produisent régulièrement du JSON approximatif. Plutôt que de
rejeter le plan entier, `planning.py` répare ce qui peut l'être sans changer le
sens :

- `repair_common_json_defects()` supprime les virgules finales et les
  commentaires `//`, et convertit `True` / `False` / `None` ;
- `extract_json_block()` tente une seconde passe sur la version réparée et
  signale explicitement une réponse coupée (accolades déséquilibrées) ;
- un `expected_output` manquant, des critères d'acceptation absents, un
  `requires_review` mal typé ou une dépendance invalide sont corrigés et
  tracés dans `PlanReport.corrections` au lieu d'invalider le plan ;
- `PlanError` porte un drapeau `semantic` : une erreur de forme déclenche une
  demande de reformatage, une erreur de contenu une demande de correction
  réelle du plan.

### Reprise automatique des sorties tronquées

Le budget de sortie de chaque modèle est déclaré dans `core/models.py`. Quand
une réponse est malgré tout coupée (`stopReason == "max_tokens"`),
`workers.invoke_model_with_continuation()` et
`workers.invoke_agent_with_continuation()` relancent le modèle avec le texte
déjà produit et une consigne de continuation, jusqu'à `MAX_CONTINUATIONS`
reprises, puis concatènent le résultat. Les métriques sont agrégées sans
double comptage.

Si la sortie reste tronquée, le livrable est explicitement préfixé
`[LIVRABLE INCOMPLET]` : le modèle suivant ne doit plus interpréter la coupure
comme une erreur d'exécution. Le Chef utilise le même mécanisme, ce qui
supprime la cause principale des plans JSON et des synthèses coupés.

### Workers

`workers.py` fournit :

- protection de toutes les étapes de `run_single_task` ;
- refus des livrables vides ;
- retries sélectifs avec backoff et jitter ;
- limite globale et limite par modèle ;
- protection de `future.result()` ;
- ordre de résultats identique à l'ordre initial ;
- tokens, latence, tentatives et erreurs ;
- annulation coopérative.

### Revue croisée

Le drapeau `requires_review` du plan a désormais un effet réel.
`ChefDeProjet.review_flagged_tasks()` soumet chaque livrable concerné à un
agent **critique indépendant**, d'un tier strictement supérieur à celui qui a
produit le livrable (ou du tier maximal si déjà au sommet).

La critique est bornée : un appel par tâche, aucun débat libre, aucune
réécriture. Le critique répond selon un protocole textuel simple
(`VERDICT: CONFORME` ou `VERDICT: A CORRIGER`, puis des défauts actionnables),
ce qui évite tout nouveau parsing JSON fragile.

- verdict conforme : la tâche reste `terminé` ;
- verdict non conforme : la tâche repasse `à revoir`, reste relançable, et son
  livrable est conservé pour la synthèse ;
- une relance efface le verdict précédent afin que le nouveau livrable soit
  recritiqué.

### Revue et synthèse

L'UI n'appelle jamais automatiquement `synthesize()` après `execute()`.
L'utilisateur examine les résultats, lance la revue croisée, relance les tâches
éligibles ou autorise une synthèse complète ou partielle.

`synthesize()` n'exige plus que toutes les tâches soient réussies : un unique
échec définitif condamnait auparavant le chantier. Elle est refusée uniquement
tant que des tâches restent exécutables (`à faire` / `en cours`). Le prompt
reçoit un bloc `ÉTAT DU CHANTIER` COMPLET ou PARTIEL, et les incidents couvrent
les tâches en échec, bloquées, annulées et à revoir.

### Tests du mode multi-agent

`tests/test_plan_preview.py` (bibliothèque standard `unittest`, aucun appel
Bedrock) couvre la cohérence de l'éditeur de plan :

- le format de description enrichie est identique à celui consommé par
  l'exécution ;
- les dépendances inconnues et les cycles sont refusés ;
- une édition invalide ne mute pas le ledger ;
- une édition valide conserve le format canonique attendu par le worker.

## Client Bedrock

- Client mis en cache par région dans `llm_engine.py`.
- Timeouts configurables : `BEDROCK_CONNECT_TIMEOUT` et
  `BEDROCK_READ_TIMEOUT`.
- Retry SDK court via `BEDROCK_SDK_TOTAL_MAX_ATTEMPTS` pour ne pas multiplier
  silencieusement les retries déjà gérés dans les workers.
- La température est normalisée par `sanitize_temperature()`.

## Tests

Suites présentes (bibliothèque standard `unittest`, aucun appel Bedrock) :

- `tests/test_plan_preview.py` : cohérence du format de description, rejet des
  dépendances inconnues ou cycliques, et absence de mutation en cas d'erreur.

Suite référencée mais absente de cette copie du dépôt :

- `tests/test_multi_agent_discovery.py` : couverture de la phase DISCOVERY
  (outils du lecteur, déduplication des appels, lecture réelle avant
  planification). À restaurer ou à réécrire.

Exécution :

```bash
python -m compileall -q core tools app.py llm_engine.py
python -m unittest discover -s tests -v
```

## Invariants

- Aucune écriture sans approbation humaine.
- Aucun client Boto3, modèle LangChain, pool, future, verrou ou sémaphore dans
  `st.session_state`. Exception assumée et unique : le `BackgroundJob` de
  l'opération en cours, qui doit être partagé entre le script et son thread.
- Le thread de travail ne touche jamais `st.session_state`.
- Aucun livrable vide ne passe à `DONE`.
- Aucun échec technique n'est transformé automatiquement en `REVIEW`.
- Aucune synthèse n'est présentée comme complète si des tâches sont en échec ou
  bloquées.
- Les contenus de fichiers, mémoires et livrables sont des données non fiables,
  jamais des instructions supérieures.
- Les informations indisponibles ne sont pas inventées.
