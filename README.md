# MTG Deck Keeper

Photograph your Magic: The Gathering cards and get back a counted, checked decklist, read by a
vision model running on your own hardware.

- **Upload several photos at once.** On a phone, the upload button opens the camera.
- **Card pre-pass (on by default, can be turned off per scan).** OpenCV cuts each card out of
  the photo before the model reads it. The whole photo is always read too, so fanned or
  overlapping cards are still found. Where the two counts disagree, the line is flagged.
- **Checked against Scryfall.** Every name is fuzzy-matched against Scryfall's full card-name
  list, so "Lightning Bolf" becomes *Lightning Bolt*. The list updates daily and only new names
  are added.
- **Review before saving.** Low-confidence and unmatched lines are highlighted with
  "did you mean" suggestions, card search, quantity buttons and main/side/commander boards.
- **Decks** grouped by card type, with a mana curve and MTG Arena or plain-text export.
- **OIDC login** with the same options and env var names as Mealie. Every setting can be set
  by environment variable *or* in the admin Settings page.
- **One Docker image** (`ghcr.io/adman234/mtg-deck-keeper`) for amd64 and arm64, plus an Unraid
  template.

## Quick start

### Unraid

```sh
wget -O /boot/config/plugins/dockerMan/templates-user/my-mtg-deck-keeper.xml \
  https://raw.githubusercontent.com/adman234/mtg-deck-keeper/main/unraid/mtg-deck-keeper.xml
```

Then **Docker → Add Container → Template: mtg-deck-keeper**. See [docs/unraid.md](docs/unraid.md).

### Docker

```sh
docker run -d --name deck-keeper -p 9925:9925 -v ./data:/app/data \
  ghcr.io/adman234/mtg-deck-keeper:latest
```

Or use [docker-compose.yml](docker-compose.yml), which also runs llama-swap.

Open `http://localhost:9925`, create the admin account, then set up the vision model under
**Settings → Vision model**. For llama-swap, see [docs/llama-swap.md](docs/llama-swap.md).

## Configuration

Settings come from three places, highest priority first:

1. **Environment variable.** It is shown locked in the GUI.
2. **Value saved in Settings.** Secrets are stored encrypted.
3. Built-in default.

| Variable | Default | Meaning |
|---|---|---|
| `VISION_ENABLED` | `true` | Turn photo scanning on/off |
| `VISION_BASE_URL` | `http://llama-swap:8080/v1` | OpenAI-compatible endpoint |
| `VISION_API_KEY` | – | Only if your server checks keys |
| `VISION_MODEL` | – | Model ID, e.g. the llama-swap key `daily` |
| `VISION_TIMEOUT` | `300` | Seconds per request (model loading included) |
| `VISION_MAX_CONCURRENCY` | `1` | Parallel requests; match `--parallel` |
| `VISION_CONFIDENCE_THRESHOLD` | `0.80` | Flag lines below this |
| `VISION_DISABLE_THINKING` | `true` | Ask thinking models to answer directly |
| `MATCH_COLLECTOR_NUMBER` | `true` | Identify cards by set code + collector number |
| `MATCH_FOREIGN_NAMES` | `true` | Search Scryfall for foreign-language and alternate names |
| `MATCH_REASK` | `true` | Re-ask the model with candidates for unclear cut-out cards |
| `PILE_ZOOM` | `true` | Re-read each pile from a close-up (count and name) |
| `PILE_COUNTS` | `false` | Count several copies stacked or fanned in one pile; off, each card found counts once |
| `TILES_WITH_COLUMNS` | `false` | Also read cascade photos in tiles (finds cards outside the columns, about twice as slow) |
| `VISION_IMAGE_MAX_SIDE` | `1600` | Whole photos are shrunk to this before sending |
| `CARD_DETECTION_DEFAULT` | `true` | Pre-pass checkbox default |
| `ALLOW_PASSWORD_LOGIN` | `true` | Username/password login |
| `OIDC_AUTH_ENABLED` | `false` | Enable OIDC |
| `OIDC_CONFIGURATION_URL` | – | `…/.well-known/openid-configuration` |
| `OIDC_CLIENT_ID` / `OIDC_CLIENT_SECRET` | – | Client credentials |
| `OIDC_SIGNUP_ENABLED` | `true` | Create users on first login |
| `OIDC_USER_GROUP` / `OIDC_ADMIN_GROUP` | – | Group gate and admin group |
| `OIDC_AUTO_REDIRECT` | `false` | Skip the login page (`/login?direct=1` bypasses) |
| `OIDC_PROVIDER_NAME` | `OAuth` | Login button text |
| `OIDC_REMEMBER_ME` | `false` | 30-day sessions for OIDC logins |
| `OIDC_USER_CLAIM` | `email` | `email`, `preferred_username` or `sub` |
| `OIDC_GROUPS_CLAIM` | `groups` | Claim holding group names |
| `OIDC_SCOPES_OVERRIDE` | – | Default `openid profile email groups` |
| `SCRYFALL_AUTO_SYNC` / `SCRYFALL_SYNC_HOUR` | `true` / `4` | Daily card-name check |

Process settings (env only): `BASE_URL` (public URL, used for the OIDC redirect URI), `PORT`
(`9925`), `PUID`/`PGID` (`99`/`100`), `DATA_DIR` (`/app/data`), `DATABASE_URL` (SQLite by default,
or `postgresql+psycopg://…`), `SECRET_KEY` (generated and saved in the data folder if unset),
`TOKEN_TIME` (session hours, `48`).

The OIDC redirect URI is `<BASE_URL>/api/auth/oidc/callback`. Settings → Login & OIDC shows the
exact value to copy into your provider.

## How a scan works

1. Photos are rotated using their EXIF data, shrunk and stored.
2. With the pre-pass on, OpenCV finds card-shaped outlines (63 × 88 mm), flattens each card and
   sends it to the model on its own. Each photo is also sent whole.
3. The model returns JSON (name, count, set, collector number, confidence) using structured output.
4. Names are fuzzy-matched against the local Scryfall list, including each face of
   double-faced cards. Cards that don't match well are looked up by set code and collector
   number, then by foreign or alternate name, and unclear cut-out cards are shown to the model
   again with candidates. See [docs/foreign-and-showcase.md](docs/foreign-and-showcase.md).
5. Within one photo, a card's quantity is the higher of the crop count and the whole-photo count.
   Across photos, quantities are added.
6. A line is flagged when its confidence (model × name match) is under the threshold, when nothing
   matched, when the two counts disagree, or when a basic land count is over 4.

## Development

```sh
# backend
python -m venv .venv && .venv/bin/pip install -e "backend[dev]"
cd backend && DATA_DIR=./data ../.venv/bin/uvicorn deck_keeper.main:app --reload --port 9925
../.venv/bin/pytest

# frontend (proxies /api to :9925)
cd frontend && npm install && npm run dev
```

The stack follows Mealie: FastAPI, SQLAlchemy 2 + Alembic (migrations run on start) and Pydantic.
The frontend is Vue 3 + Vite, built into the backend's `static/` folder. To add a migration, run
`python -m deck_keeper.migrate "message"`.

## Roadmap

See the [project plan](https://claude.ai/artifact/FcqoXo42nKZ3VXPRryJ3rr). Phase 0 (skeleton) and
Phase 1 (scan to deck) are done; OIDC and the settings GUI from Phase 2 are also in.
Options for foreign-language and showcase cards are in
[docs/foreign-and-showcase.md](docs/foreign-and-showcase.md).

## Legal

Licensed under AGPL-3.0-or-later. Card names and data come from [Scryfall](https://scryfall.com).
Magic: The Gathering is © Wizards of the Coast. This project is unofficial Fan Content and is not
approved or endorsed by Wizards.
