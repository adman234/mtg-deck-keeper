# Working on MTG Deck Keeper

## Workflow

- When the owner asks for a change, finish it end to end: run the checks below, commit and push
  straight to `main`. No pull requests and no feature branches; don't leave work on a branch.
- After pushing, watch the CI run on `main`; if it fails, fix it and push again.

## Checks to run before pushing

- Backend: `cd backend && pytest -q` (install with `pip install -e "backend[dev]"`)
- Frontend: `cd frontend && npm ci && npm run build`

## Layout

- `backend/deck_keeper/` FastAPI app; runtime settings live in `app_settings.py`
  (env var > GUI value > default). New settings go in `SPECS` and appear in the GUI automatically.
- `backend/deck_keeper/services/` vision client, OpenCV detection, Scryfall sync, matching, scan worker.
- `frontend/src/` Vue 3 + Vite, built into `backend/deck_keeper/static/`.
- `unraid/`, `Dockerfile`, `.github/workflows/docker.yml` publish `ghcr.io/adman234/mtg-deck-keeper`.
- Add a migration with `python -m deck_keeper.migrate "message"` after changing `models.py`.
