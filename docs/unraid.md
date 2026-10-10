# Installing on Unraid

1. Open a terminal on the server (Unraid web UI → >_ icon) and run:

   ```sh
   wget -O /boot/config/plugins/dockerMan/templates-user/my-mtg-deck-keeper.xml \
     https://raw.githubusercontent.com/adman234/mtg-deck-keeper/main/unraid/mtg-deck-keeper.xml
   ```

2. Go to **Docker → Add Container**, pick **mtg-deck-keeper** from the *Template* list.
3. Check the port (default 9925) and app data path, then click **Apply**. Every other field is
   optional. Anything you leave blank can be set later in the web UI under **Settings**.
   Anything you fill in here is shown locked in the web UI.
4. Open `http://<unraid-ip>:9925` and create the admin account.
5. In **Settings → Vision model**, enter your llama-swap address, e.g.
   `http://<unraid-ip>:8080/v1`, click **Test connection & list models**, pick the model and save.

Updates: the template uses `:latest`, so Unraid's **Check for updates** picks up new releases.

The container runs as `PUID=99` / `PGID=100` (nobody:users) and fixes ownership of the app data
folder on start.
