# Using llama-swap

Deck Keeper talks to any OpenAI-compatible server. [llama-swap](https://github.com/mostlygeek/llama-swap)
is a good fit: it starts `llama-server` on demand and swaps models for you.

## Example config

```yaml
macros:
  # llama-swap fills in ${PORT}. Without --port ${PORT} it can't reach the model.
  llama: /app/llama-server --host 127.0.0.1 --port ${PORT}

models:
  daily:                       # <- this key is the model ID Deck Keeper uses
    name: "Defiant 9B"         # display name in llama-swap's UI only
    cmd: |
      ${llama}
      -m /models/defiant-9b.gguf
      --mmproj /models/defiant-9b-mmproj.gguf
      --image-min-tokens 1024
      -c 32768
      --parallel 3
      --cache-reuse 256
      --cache-ram 4096
      --kv-unified
      --reasoning-budget 0     # only if the model "thinks"; see below
```

Then in Deck Keeper (Settings → Vision model, or env vars):

| Setting | Value |
|---|---|
| API base URL | `http://<llama-swap host>:8080/v1` |
| Model | `daily` (the key, not the `name`) |
| Parallel requests | `3`, to match `--parallel 3` |
| Request timeout | leave at 300 s so the first request can wait for the model to load |

Use **Test connection & list models** to confirm Deck Keeper can see `daily`.

## Notes on the flags

- **Thinking models** (Qwen3 and similar fine-tunes) reason before answering. If the reasoning
  uses up the reply budget, the answer comes back empty and the scan fails with
  "Model reply had no JSON". Deck Keeper asks the server to skip thinking on every request
  ("Turn off model thinking" in Settings, on by default). If your model's chat template ignores
  that, add `--reasoning-budget 0` to its llama-server command. Card reading doesn't benefit
  from reasoning, and it makes each request much slower.

- `--mmproj` is what makes the model see images. Without it the server rejects image input.
- `--image-min-tokens 1024` gives each image more tokens, which helps with small title text.
- With `--kv-unified` the three parallel slots share the 32k context. One request uses about
  1–3k tokens (image + prompt + reply), so there is plenty of room.
- Deck Keeper sends `response_format` with a JSON schema. llama-server turns that into a grammar,
  so replies are always valid JSON. If a server rejects it, Deck Keeper retries without it.
- `--cache-reuse` / `--cache-ram` let the shared system prompt be reused between requests.
