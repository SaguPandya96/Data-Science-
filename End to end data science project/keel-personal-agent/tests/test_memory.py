from __future__ import annotations

from datetime import timedelta

import pytest

from keel.memory.retrieval import KeelMemory, LexicalMemory, RecentMemory, bm25_scores
from keel.memory.text import canonical_key, expand, stem, tokens


def test_canonical_key_merges_spelling_variants_only():
    assert canonical_key("Home City") == canonical_key("user_home-city") == "home_city"
    assert canonical_key("current_employer") == "employer"
    assert canonical_key("residence") != canonical_key("home_city")
    assert canonical_key(None) is None


def test_stem_is_consistent_across_word_forms():
    # Both bugs the hand-written stemmer had: "lives"/"live" and "siblings"/"sibling".
    assert stem("lives") == stem("live")
    assert stem("siblings") == stem("sibling")
    assert stem("named") == stem("name")
    assert tokens("Where do I live these days?") == ["live"]


def test_expand_adds_weighted_synonyms():
    weights = expand("my partner")
    assert weights[stem("partner")] == 1.0
    assert weights[stem("spouse")] == 0.6


def test_new_value_under_same_key_supersedes_old(store, clock):
    old, _ = store.add("Lives in Denver.", kind="fact", key="home_city")
    clock.advance(days=10)
    new, replaced = store.add("Moved to Austin.", kind="fact", key="Home City")
    assert [m.id for m in replaced] == [old.id]
    assert store.get(old.id).superseded_by == new.id
    assert [m.text for m in store.all()] == ["Moved to Austin."]
    assert [m.id for m in store.history(new.id)] == [old.id]


def test_different_key_does_not_supersede(store):
    store.add("Lives in Denver.", key="home_city")
    _, replaced = store.add("Moved to Austin.", key="residence")
    assert replaced == []
    assert len(store.all()) == 2


def test_forget_hides_and_blanks_memory(store):
    memory, _ = store.add("Secret-ish thing.", key="x")
    store.forget(memory.id)
    assert store.all() == []
    assert store.get(memory.id).text == "[forgotten]"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"kind": "rumor"}, "kind"), ({"importance": 9}, "importance")],
)
def test_add_validates(store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.add("text", **kwargs)


def test_bm25_prefers_matching_document():
    scores = bm25_scores(["the cat sat", "dogs run fast", "a cat and a dog"], {"cat": 1.0})
    assert scores[0] > scores[1] == 0
    assert scores[2] > 0


def _history(store, clock):
    store.add("Severely allergic to peanuts.", kind="constraint", key="allergy")
    clock.advance(days=1)
    store.add("Lives in Denver.", kind="fact", key="home_city")
    clock.advance(days=1)
    store.add("Adopted a parrot named Kiwi.", kind="fact", key="pet")
    for _ in range(4):
        clock.advance(hours=1)
        store.add("Asked how to clean a car's interior.", kind="episode")
    clock.advance(days=30)
    store.add("Moved to Austin.", kind="fact", key="home_city")
    return store.all(include_inactive=True)


def test_keel_excludes_superseded_and_pins_constraints(store, clock):
    memories = _history(store, clock)
    chosen = KeelMemory().retrieve(memories, "Which city do I live in?", clock.now(), 3)
    texts = [m.text for m in chosen]
    assert texts[0] == "Severely allergic to peanuts."
    assert "Moved to Austin." in texts
    assert "Lives in Denver." not in texts


def test_lexical_baseline_shows_stale_value(store, clock):
    memories = _history(store, clock)
    chosen = LexicalMemory().retrieve(memories, "Where do I live?", clock.now(), 3)
    assert "Lives in Denver." in [m.text for m in chosen]


def test_key_indexing_finds_memory_that_never_names_its_topic(store, clock):
    memories = _history(store, clock)
    with_keys = KeelMemory(core_max=0).retrieve(memories, "What pet do I have?", clock.now(), 1)
    assert with_keys[0].text == "Adopted a parrot named Kiwi."


def test_duplicate_suppression(store, clock):
    memories = _history(store, clock)
    query = "How do I clean my car?"
    plain = KeelMemory(core_max=0, use_dedupe=False).retrieve(memories, query, clock.now(), 4)
    deduped = KeelMemory(core_max=0).retrieve(memories, query, clock.now(), 4)
    assert sum(m.text.startswith("Asked how to clean") for m in plain) > 1
    assert sum(m.text.startswith("Asked how to clean") for m in deduped) == 1


def test_constraints_never_take_more_than_half_of_k(store, clock):
    for allergen in ("peanuts", "shellfish", "sesame"):
        store.add(f"Allergic to {allergen}.", kind="constraint", key=f"allergy_{allergen}")
    store.add("Lives in Austin.", key="home_city")
    memories = store.all()
    assert all(
        m.kind != "constraint" for m in KeelMemory().retrieve(memories, "city", clock.now(), 1)
    )
    at_five = KeelMemory().retrieve(memories, "city", clock.now(), 5)
    assert sum(m.kind == "constraint" for m in at_five) >= 2


def test_recent_is_newest_first(store, clock):
    for i in range(5):
        clock.advance(minutes=1)
        store.add(f"Note {i}.", kind="episode")
    chosen = RecentMemory().retrieve(store.all(), "anything", clock.now() + timedelta(days=1), 2)
    assert [m.text for m in chosen] == ["Note 4.", "Note 3."]
