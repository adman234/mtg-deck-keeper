# Foreign-language and showcase cards: options

Right now the model is asked for the English name if it recognises the card, and the result is
fuzzy-matched against Scryfall's English name list. That works for most English cards, including
many showcase frames, but it breaks down for:

- **Non-English printings**, especially Japanese, Chinese, Korean and Russian titles.
- **Stylised showcase titles** that small models misread.
- **Cards with a different printed name**, such as Godzilla series and Universes Within cards
  (Scryfall calls this `flavor_name`).

Possible approaches, roughly cheapest first. Several combine well.

| # | Approach | Handles | Cost | Needs internet at scan time |
|---|---|---|---|---|
| 1 | **Read the set code and collector number** from the bottom-left corner and look up that exact printing (`/cards/{set}/{number}`). The model already returns these when legible. | Any language, showcase, borderless, alternate names; also gives the exact printing for pricing later | Small. Cards before 2015 (M15 frame) have no collector line | Yes, or a cached printing table |
| 2 | **Ask Scryfall about unmatched names**: search `name:"…" lang:any` with `include_multilingual=true`, then cache the answer. | Foreign names in any script | Small. One request per unmatched card | Yes |
| 3 | **Re-ask the model with candidates**: for a low-confidence crop, send only the title strip plus the top 5 fuzzy candidates and ask it to pick one. | Stylised showcase fonts, glare, bad crops | One extra model call per flagged card | No |
| 4 | **Local multilingual name index**: download Scryfall's `all_cards` bulk file once (~2 GB), keep only `printed_name` → English name (a few MB), and refresh it weekly. Add `flavor_name`s too. | Foreign names and alternate names, fully offline | Large first download; weekly refresh (Scryfall has no change feed) | No |
| 5 | **Match card art**: store perceptual hashes or CLIP embeddings of Scryfall artwork and compare against each crop. | Any language, any frame, even unreadable titles | Several GB of images and a second model; most work to build | No |

## Suggestion

**1 + 2 + 3** covers almost everything at low cost. Use the collector number when it's readable,
fall back to Scryfall search for foreign names, and use the candidate re-ask for stylised titles.
Add **4** later if fully offline scanning matters. **5** is the most robust but the most work, so
it suits a later phase.
