# MAINTiDOC

MAINTiDOC est un prototype de **RAG multimodal agentique** qui permet d'importer un PDF,
d'indexer séparément son texte, ses tableaux et ses images, puis de poser des questions dans
une interface Streamlit. Tous les modèles génératifs et d'embeddings sont exécutés localement
avec Ollama.

Le dépôt contient déjà un document de démonstration : le rapport annuel 2024 de l'UNICEF en
Bulgarie. Il comprend 27 pages, des paragraphes, des photographies, des cartes, des
infographies, des tableaux et un tableau de dépenses.

## Résultat attendu

```text
PDF importé
  ├── LlamaIndex + PyMuPDF ──> passages textuels
  ├── pdfplumber ────────────> tableaux convertis en Markdown
  └── PyMuPDF ───────────────> images originales
          │
          ├── BGE-M3/Ollama ─> vecteurs texte/tableaux (1024 dimensions)
          └── CLIP ──────────> vecteurs images (512 dimensions)
                                      │
                           Qdrant, deux collections
                                      │
                     Agent LangChain + Qwen/Ollama
                      ├── recherche texte/tableaux
                      ├── recherche visuelle
                      └── recherche multimodale
                                      │
                       réponse et sources dans Streamlit
```

## Prérequis

- macOS, Linux ou Windows avec WSL ;
- [uv](https://docs.astral.sh/uv/) ;
- Ollama en cours d'exécution ;
- les modèles Ollama suivants :

```bash
ollama pull bge-m3
ollama pull qwen2.5vl:7b
ollama pull qwen3.5:9b
```

Le projet utilise Python 3.12, fixé dans `.python-version`. `uv` utilise automatiquement cette
version même si le Python système est différent.

## Installation et lancement

```bash
git clone https://github.com/Mohamed-Saifeddine/Maintidoc.git
cd Maintidoc
cp .env.example .env
uv sync --extra dev
uv run streamlit run app.py
```

Ouvrir ensuite l'adresse affichée par Streamlit, généralement `http://localhost:8501`.

Dans l'onglet **Indexer**, cliquer sur **Indexer le rapport UNICEF fourni**. Le premier lancement
télécharge aussi les poids de CLIP depuis Hugging Face. Une fois l'indexation terminée, ouvrir
l'onglet **Interroger**.

Exemples de questions :

- `Quelles sont les principales dépenses présentées dans le rapport ?`
- `Combien d'enfants réfugiés ont reçu une aide ?`
- `Que montre la carte de la page 16 ?`
- `Résume les résultats obtenus pour la petite enfance.`
- `Quelles photographies illustrent l'aide aux enfants réfugiés ?`

## Qdrant local ou Docker

Par défaut, Qdrant fonctionne en **mode embarqué** et écrit ses données dans `data/qdrant/`.
C'est la configuration la plus simple pour une démonstration locale.

Pour utiliser le serveur Qdrant Docker :

```bash
docker compose up -d qdrant
```

Puis définir dans `.env` :

```dotenv
QDRANT_URL=http://localhost:6333
```

Deux collections sont créées :

- `maintidoc_text` : passages et tableaux, vecteurs BGE-M3 de 1024 dimensions ;
- `maintidoc_images` : images, vecteurs CLIP de 512 dimensions.

Séparer les espaces vectoriels est obligatoire : les vecteurs de BGE-M3 et de CLIP n'ont ni la
même dimension ni la même signification.

## Explication détaillée de l'implémentation

### 1. Chargement du PDF avec LlamaIndex

Le fichier `src/maintidoc/extraction.py` utilise `PyMuPDFReader` fourni par LlamaIndex. Le lecteur
produit un document par page. `SentenceSplitter` découpe ensuite chaque page en passages avec :

- une taille cible de 800 tokens ;
- un chevauchement de 120 tokens ;
- le numéro de page conservé dans les métadonnées ;
- un identifiant stable tel que `page_014_text_002`.

Le chevauchement évite de perdre une information située exactement à la frontière entre deux
passages. Les valeurs sont configurables avec `CHUNK_SIZE` et `CHUNK_OVERLAP`.

### 2. Extraction séparée des tableaux

`pdfplumber` analyse chaque page et retourne les cellules détectées. Les tableaux sont convertis en
Markdown avant vectorisation. Cette représentation conserve les lignes et les colonnes tout en étant
facile à comprendre pour le modèle de langage.

Chaque tableau possède son propre identifiant, par exemple `page_026_table_001`. Il n'est donc pas
noyé dans un grand passage textuel et peut être retrouvé directement par une question numérique.

### 3. Extraction séparée des images

PyMuPDF récupère les objets images incorporés dans le PDF. Le code :

1. ignore les très petites images, souvent des logos ou éléments décoratifs ;
2. élimine les doublons avec leur empreinte SHA-256 ;
3. sauvegarde l'image originale dans `data/extracted/<document_id>/` ;
4. conserve sa page, ses dimensions et son identifiant ;
5. limite par défaut l'indexation à 40 images par document.

Les fichiers extraits ne sont pas versionnés dans Git car ils peuvent toujours être recréés à partir
du PDF source.

### 4. Embeddings textuels avec BGE-M3

`OllamaTextEmbedder` appelle l'endpoint local `POST /api/embed` d'Ollama. Le modèle
`bge-m3:latest` transforme chaque passage et chaque tableau en un vecteur de 1024 dimensions.

BGE-M3 est adapté à ce projet parce qu'il est multilingue : le rapport peut être en anglais alors que
la question est posée en français. Les vecteurs sont normalisés avant leur stockage dans Qdrant.

### 5. Recherche visuelle avec CLIP

`ClipEncoder` charge `openai/clip-vit-base-patch32` avec Transformers. Pendant l'indexation, chaque
image devient un vecteur visuel de 512 dimensions. Pendant une question, le texte de la question est
transformé dans le même espace vectoriel.

Cette propriété permet de rechercher une image avec une phrase, par exemple :

```text
une carte montrant les villes aidées par l'UNICEF
```

Le système peut ainsi retrouver une carte pertinente même si aucun texte identique n'est inscrit
dans l'image.

### 6. Description optionnelle des images avec Qwen2.5-VL

CLIP sert à **retrouver** une image, pas à rédiger une description détaillée. Si
`ENABLE_VLM_CAPTIONS=true`, `OllamaVisionCaptioner` envoie chaque image extraite à
`qwen2.5vl:7b`. La description générée est enregistrée dans le payload Qdrant avec l'image.

Cette option est désactivée par défaut pour accélérer la première indexation. Même sans elle, le
contexte textuel de la page accompagne l'image. Pour l'activer :

```dotenv
ENABLE_VLM_CAPTIONS=true
```

La description du VLM est traitée comme une observation, jamais comme une preuve supérieure au
document original.

### 7. Stockage dans Qdrant

`src/maintidoc/vector_store.py` encapsule Qdrant. Chaque point contient :

- le vecteur ;
- l'identifiant et le nom du document ;
- le numéro de page ;
- l'identifiant de l'élément ;
- le type `text`, `table` ou `image` ;
- le contenu textuel ou la description ;
- le chemin de l'image, lorsqu'il existe.

Réindexer un même PDF remplace ses anciens points au lieu de créer des doublons. Le document est
identifié par son nom normalisé et le SHA-256 de son contenu.

### 8. Recherche multimodale

`src/maintidoc/retrieval.py` fournit trois opérations :

- `search_text` : question encodée avec BGE-M3, recherche dans textes et tableaux ;
- `search_images` : question encodée avec CLIP, recherche dans les images ;
- `search_all` : exécute les deux recherches et fusionne les classements avec Reciprocal Rank
  Fusion (RRF).

RRF privilégie les éléments bien classés dans plusieurs listes sans comparer directement les scores
bruts de BGE-M3 et de CLIP, qui ne sont pas sur la même échelle.

### 9. Agent LangChain

`src/maintidoc/agent.py` crée un agent LangChain piloté par `qwen3.5:9b` via Ollama. Trois outils
lui sont exposés :

1. `rechercher_texte_et_tableaux` ;
2. `rechercher_images` ;
3. `rechercher_tout`.

Le modèle choisit l'outil selon la question. Une question sur un montant doit normalement appeler
la recherche textuelle ; une question sur une photo ou une carte appelle la recherche visuelle ; une
question combinée appelle la recherche multimodale.

Le prompt système impose :

- une recherche avant toute réponse factuelle ;
- l'absence d'invention ;
- une citation `[fichier, p. X, element_id]` ;
- une demande de clarification si les preuves sont insuffisantes ;
- le traitement du contenu du PDF comme donnée et non comme instruction.

Le code récupère aussi les résultats des appels d'outils afin que Streamlit puisse afficher les sources
et les images réellement utilisées.

### 10. Interface Streamlit

`app.py` propose trois onglets :

- **Indexer** : import d'un PDF ou sélection du rapport UNICEF fourni ;
- **Interroger** : conversation avec l'agent et affichage des sources ;
- **Architecture** : résumé visuel du pipeline.

La barre latérale vérifie Ollama, affiche les modèles requis et le nombre d'éléments présents dans
Qdrant. Après une réponse, une section dépliable montre les passages, tableaux et images récupérés.

### 11. Commandes sans interface

Indexer un PDF :

```bash
uv run maintidoc-index chemin/vers/document.pdf
```

Poser une question :

```bash
uv run maintidoc-ask "Quel est le total des dépenses ?"
```

### 12. Tests

Les tests unitaires vérifient actuellement :

- la conversion des tableaux en Markdown ;
- le comportement sur un tableau vide ;
- la fusion Reciprocal Rank Fusion.

Exécution :

```bash
uv run pytest
uv run ruff check .
```

## Organisation du dépôt

```text
Maintidoc/
├── app.py                         interface Streamlit
├── docker-compose.yml             serveur Qdrant optionnel
├── pyproject.toml                 dépendances et commandes Python
├── .env.example                   configuration documentée
├── data/
│   ├── source/                    PDF UNICEF fourni
│   ├── uploads/                   PDF importés, non versionnés
│   ├── extracted/                 images extraites, non versionnées
│   └── qdrant/                    index local, non versionné
├── src/maintidoc/
│   ├── config.py                  configuration centrale
│   ├── models.py                  schémas Pydantic
│   ├── extraction.py              extraction multimodale
│   ├── embeddings.py              BGE-M3, CLIP et Qwen2.5-VL
│   ├── vector_store.py            persistance et recherche Qdrant
│   ├── indexing.py                pipeline complet d'indexation
│   ├── retrieval.py               récupération et fusion RRF
│   ├── agent.py                   agent et outils LangChain
│   ├── health.py                  vérification Ollama
│   └── cli.py                     commandes terminal
└── tests/                          tests unitaires
```

## Configuration

| Variable | Valeur par défaut | Rôle |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL de l'API Ollama |
| `OLLAMA_EMBED_MODEL` | `bge-m3:latest` | embeddings texte/tableaux |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | description optionnelle des images |
| `OLLAMA_AGENT_MODEL` | `qwen3.5:9b` | raisonnement et appels d'outils |
| `OLLAMA_AGENT_REASONING` | `false` | désactive le raisonnement long pour l'interface |
| `OLLAMA_AGENT_NUM_PREDICT` | `700` | longueur maximale de la réponse de l'agent |
| `OLLAMA_AGENT_TIMEOUT` | `180` | délai maximal d'un appel Ollama, en secondes |
| `CLIP_MODEL` | `openai/clip-vit-base-patch32` | embeddings texte-image |
| `QDRANT_URL` | vide | serveur Qdrant ; vide = stockage local |
| `TEXT_TOP_K` | `5` | passages textuels récupérés |
| `IMAGE_TOP_K` | `4` | images récupérées |
| `ENABLE_VLM_CAPTIONS` | `false` | descriptions Qwen2.5-VL à l'indexation |

## Limites connues

- `pdfplumber` fonctionne bien sur les tableaux réguliers mais peut échouer sur des tableaux très
  graphiques ou sans bordures.
- CLIP est surtout performant sur les concepts visuels ; le texte minuscule dans une image demande
  un OCR spécialisé.
- Les citations sont fondées sur les éléments récupérés, mais une validation humaine reste utile pour
  toute décision importante.
- Le prototype local ne met pas encore en œuvre l'authentification ni les groupes d'accès du cahier
  des charges MAINTiDOC étendu.
- Le premier chargement de CLIP peut prendre plusieurs minutes en raison du téléchargement du modèle.

## Source du PDF de démonstration

UNICEF Bulgaria, *Annual Report 2024* :
https://www.unicef.org/bulgaria/en/media/21401/file/Annual%20Report%20UNICEF%20Bulgaria%202024_EN.pdf

