# crowe-ml-pipeline

Python monorepo from 2025 for fungal compound screening: a scikit-learn and RDKit analysis package (universal-fungal-intelligence-system) with a Streamlit UI, a Kaggle notebook, two single-file agents, and Fly, Cloud Build and GKE deploy files; the pinned dependency `crowechem` is not on PyPI, so the root requirements cannot resolve.

## Status

experimental

36 commits between 2025-06-28 and 2025-09-14 (`git log`), nothing since. On 2026-09-11 one script ran (`build_dataset.py`, through its built-in fallback) and nothing else was installed, so the repository is kept as experimental rather than archived. Development stopped in September 2025.

What does not work:

- `requirements.txt` pins `crowechem`, which does not exist on PyPI. `uv pip compile requirements.txt` exits 1: "crowechem was not found in the package registry".
- `pyproject.toml` (Poetry) has no lockfile and lists `torch` and `transformers`. It was not installed on the machine used today, which does not install torch. Its `packages` entries (`universal_fungal_intelligence_system`, `crios`, `ml_enhancements`) do not match the directories (`universal-fungal-intelligence-system`, `crios/src`, `ml-enhancements`), so `poetry build` has nothing to package.
- `universal-fungal-intelligence-system/requirements.txt` also lists `torch`; its six test files were not run for the same reason.
- CI: `Validate` (`validate.yaml`) failed on all 6 runs. Its matrix is written `[3.7, 3.8, 3.9, 3.10, 3.11]` without quotes, so YAML reads `3.10` as `3.1`; the `lint-test (3.1)` job fails and fail-fast cancels the rest. `Deploy to Fly.io` (`fly-deploy.yml`) failed on all 5 runs at the `test` job; no deploy step ran. `google.yml` is the GitHub GKE sample with `TODO` placeholders and a branch filter of the literal string `"main"` (with quotes); it has never run. CodeQL default setup ran 32 scheduled scans between 2025-07-27 and 2026-03-08, all failed.
- Hosted: `crowe-ml-pipeline.fly.dev` and `crowe-vision.fly.dev` (the two Fly apps named in the two `fly.toml` files) do not resolve.
- `fly-startup.sh` runs `npm start` in the vision directory, which holds only a `Dockerfile` and `fly.toml`; there is no `package.json`. The root `Dockerfile` is empty (0 bytes).
- `crowe-coder/src/index.ts` imports `@anthropic-ai/sdk` and `axios` with no `package.json` or lockfile beside it.
- Dependabot PR #8 (black bump, 2025-08-20) is open.
- No LICENSE file, though the old README linked to one.

## Install and first run

Run on 2026-09-11 with Python 3.13.14 (system) and uv:

    uv pip compile --python-version 3.11 requirements.txt
    No solution found when resolving dependencies:
    Because crowechem was not found in the package registry and you require crowechem,
    we can conclude that your requirements are unsatisfiable.

    python3 build_dataset.py --output-dir /tmp/cmp-data
    Dataset written to /tmp/cmp-data/crowechem_dataset.jsonl

The file written contains one row, `{"id": 1, "molecule": "example", "property": 0.0}`, because the script falls back to a stub when `crowechem` is missing. That is the only thing that ran.

Not run, and why: `poetry install` (no lockfile; declares torch), `pip install -r universal-fungal-intelligence-system/requirements.txt` (declares torch), `pytest` in that package (its modules import rdkit, which was not installed), `run_web_ui.py` (Streamlit, same dependencies), `upload_metrics_to_bigquery.sh` (writes to GCP project `crowechem-fungi`), `deploy.sh`, `cloudbuild.yaml`, `Dockerfile.fly`, and the Kaggle notebook.

## What runs today

- `build_dataset.py --help` and the stub run above (standard library only).
- `crios/src/crios.py` (777 lines) parses with `ast.parse`; it was not executed.

For reference, the tree contains:

- `universal-fungal-intelligence-system/`: 91 files. `src/core`, `src/analysis`, `src/ml/models`, `src/data/collectors` (PubChem, MycoBank, NCBI references), `src/web_ui` (Streamlit), `src/api/routes`, `tests/unit` (3 files), `tests/integration` (2 files), `setup.py` (version 0.1.0), `pytest.ini`, its own README and deploy guides.
- `Crowe Logic XL`: a Jupyter notebook saved as JSON without the `.ipynb` extension (Kaggle export, 2025-08-13).
- `crios/src/crios.py`, `crowe-coder/src/index.ts`, `ml-enhancements/dataset_curator.py`: single-file programs.
- `Cl/`: empty.
- Deploy files: `fly.toml` (app `crowe-ml-pipeline`, region `sjc`, builds `Dockerfile.fly` with Poetry on Python 3.10), a second `fly.toml` for app `crowe-vision` in the `crowe-vision-platform` directory, `cloudbuild.yaml`, `deploy.sh`, `scripts/*.sh`, `.devcontainer/`.
- Eight markdown documents in the root (`ARCHITECTURE_VISION.md`, `COMPREHENSIVE_FEATURES.md`, `PROJECT_STATUS.md`, `RUN_ALL_FEATURES.md`, and others).

## Roadmap

- Replace `crowechem` in `requirements.txt` with a source that exists, or vendor it.
- Quote the Python versions in `validate.yaml` so `3.10` survives YAML parsing.
- Commit a lockfile for the Poetry project, or drop `pyproject.toml`.
- Add a LICENSE file or remove the MIT badge and link.

## Limits

- Not a product and not a research result. Nothing here has been shown to find, rank or validate any compound. The old README's wording ("breakthrough therapeutics", "novel therapeutic compounds", "therapeutic potential", "drug-likeness assessment", ">85% accuracy on bioassay data") and its superlatives about scope have no dataset, trained model, evaluation log or publication in this repository. Treat them as withdrawn.
- Do not use any output of this code for medical, dietary, safety or regulatory decisions, or as evidence that a fungal compound has any effect.
- `upload_metrics_to_bigquery.sh` hard-codes GCP project `crowechem-fungi` and bucket `crowechem-fungi-ml-metrics`. `google.yml` carries the sample values `my-project`, `cluster-1`, `gke-test`. These are identifiers and placeholders, not secrets.
- Where credentials are supplied, data goes to Google Cloud (BigQuery, Storage), Fly.io, and, through `crowe-coder`, to Anthropic and a Qwen endpoint. No data handling promise is made.
- The old README described "Crowe Logic" as an engine and "Mycelium EI" as an ecosystem. Neither is code in this repository.

## License and contact

No license file. The old README and `pyproject.toml` say MIT; no `LICENSE` file backs it.

Contact: michael@crowelogic.com
