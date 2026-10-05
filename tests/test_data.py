from btt.data import scan, speaker_split, write_synthetic_corpus
from btt.labels import CLASSES


def test_split_is_speaker_disjoint(tmp_path):
    rav, cre = write_synthetic_corpus(tmp_path)
    items = scan(rav, cre)
    assert items
    sp = speaker_split(items, seed=0)
    sets = {k: set(v) for k, v in sp.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    assert all(i.label in CLASSES for i in items)
    assert {s.split("_")[0] for s in sets["test"]} == {"cremad", "ravdess"}  # both corpora held out


def test_parsers():
    from btt.data import parse_cremad, parse_ravdess

    c = parse_cremad("/x/1001_DFA_ANG_XX.wav")
    assert c and c.label == "angry" and c.speaker == "cremad_1001"
    assert parse_cremad("/x/1001_DFA_NEU_XX.wav").label == "neutral"
    r = parse_ravdess("/x/03-01-05-01-02-01-07.wav")  # emotion 05 = angry, actor 07
    assert r and r.label == "angry" and r.speaker == "ravdess_07"
    assert parse_ravdess("/x/03-01-02-01-02-01-07.wav") is None  # calm is dropped
    assert parse_ravdess("/x/03-01-08-01-02-01-07.wav") is None  # surprised is dropped
