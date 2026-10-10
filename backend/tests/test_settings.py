def test_gui_values_saved_and_secrets_masked(admin):
    r = admin.put("/api/admin/settings", json={"values": {
        "vision.base_url": "http://gpu-box:8080/v1", "vision.model": "daily", "vision.api_key": "sk-123"}})
    assert r.status_code == 200, r.text
    by_key = {s["key"]: s for s in r.json()["settings"]}
    assert by_key["vision.model"]["value"] == "daily" and by_key["vision.model"]["source"] == "db"
    assert by_key["vision.api_key"]["value"] is None and by_key["vision.api_key"]["is_set"] is True

    # None keeps a secret, "" clears it
    admin.put("/api/admin/settings", json={"values": {"vision.api_key": None}})
    assert {s["key"]: s for s in admin.get("/api/admin/settings").json()["settings"]}["vision.api_key"]["is_set"]
    admin.put("/api/admin/settings", json={"values": {"vision.api_key": ""}})
    assert not {s["key"]: s for s in admin.get("/api/admin/settings").json()["settings"]}["vision.api_key"]["is_set"]


def test_env_var_wins_and_locks_the_field(admin, monkeypatch):
    monkeypatch.setenv("VISION_MODEL", "from-env")
    by_key = {s["key"]: s for s in admin.get("/api/admin/settings").json()["settings"]}
    assert by_key["vision.model"]["value"] == "from-env" and by_key["vision.model"]["locked"]
    r = admin.put("/api/admin/settings", json={"values": {"vision.model": "gui"}})
    assert r.status_code == 400 and "VISION_MODEL" in r.json()["detail"]


def test_settings_need_admin(admin):
    admin.post("/api/users", json={"username": "bob", "email": "bob@example.com", "password": "password123"})
    admin.post("/api/auth/logout")
    admin.post("/api/auth/login", json={"username": "bob", "password": "password123"})
    assert admin.get("/api/admin/settings").status_code == 403
