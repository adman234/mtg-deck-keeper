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


def test_the_first_words_of_one_card_name_match_it():
    idx = CardIndex([(1, "Pia and Kiran Nalaar"), (2, "Siege-Gang Commander"), (3, "Siege-Gang Lieutenant"),
                     (4, "Genesis"), (5, "Genesis Chamber")])
    assert idx.match("Pia and Kiran")[0].name == "Pia and Kiran Nalaar"
    assert idx.match("Siege-Gang")[0].score < 88  # two cards start so: can't tell which
    assert idx.match("Genesis")[0].name == "Genesis"  # a whole name wins
    assert all(m.score < 88 for m in idx.match("Pia"))  # too little of the name


def test_a_legendary_card_found_by_its_title_after_the_comma():
    idx = CardIndex([(1, "Azusa, Lost but Seeking"), (2, "Elspeth, Sun's Champion"),
                     (3, "Gisela, the Broken Blade"), (4, "Bruna, the Fading Light")])
    assert idx.match("Elspeth, Lost but Seeking")[0].name == "Azusa, Lost but Seeking"
    assert idx.match("Elspeth, Sun's Champion")[0].name == "Elspeth, Sun's Champion"  # exact still wins
