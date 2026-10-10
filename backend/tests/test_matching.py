from deck_keeper.services.matching import CardIndex, normalize

INDEX = CardIndex([(1, "Lightning Bolt"), (2, "Lightning Strike"), (3, "Delver of Secrets // Insectile Aberration"),
                   (4, "Æther Vial"), (5, "Jace, the Mind Sculptor")])


def test_exact_match_scores_100():
    assert INDEX.match("Lightning Bolt")[0].score == 100


def test_ocr_typo_matches():
    m = INDEX.match("Lightning Bolf")[0]
    assert (m.card_id, m.name) == (1, "Lightning Bolt") and m.score > 85


def test_front_face_resolves_to_full_dfc_name():
    m = INDEX.match("Delver of Secrets")[0]
    assert m.name == "Delver of Secrets // Insectile Aberration" and m.score == 100


def test_accents_and_punctuation_ignored():
    assert INDEX.match("Aether Vial")[0].card_id == 4
    assert INDEX.match("jace the mind sculptor")[0].card_id == 5
    assert normalize("Æther  Vial!") == "aether vial"


def test_search_prefers_prefix():
    names = [m.name for m in INDEX.search("light")]
    assert names[:2] == ["Lightning Bolt", "Lightning Strike"]
