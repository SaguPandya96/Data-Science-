"""Synthetic users with a memory history, for the retrieval benchmark.

Each persona has 16 personal details. A detail is stated in one weekly session, may change
value in a later one, and is asked about at the end. Small-talk memories fill every session;
many share words with the questions on purpose ("asked for restaurants in Denver" while the
user lives in Austin), because real histories are full of near misses.

Question wording comes in two sets per detail: ``dev`` for building and debugging the
retriever, ``test`` for the single pre-registered run (docs/ANALYSIS_PLAN.md).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from keel.clock import FixedClock
from keel.db import connect
from keel.memory.store import Memory, MemoryStore


@dataclass(frozen=True)
class Slot:
    key: str
    alt_key: str  # what a model might call it instead; used by the key-noise arm
    kind: str
    values: tuple[str, ...]
    statements: tuple[str, ...]  # first time the detail is stated
    updates: tuple[str, ...]  # how a change is recorded
    dev: tuple[str, ...]
    test: tuple[str, ...]
    updatable: bool = True


SLOTS: tuple[Slot, ...] = (
    Slot(
        key="home_city",
        alt_key="residence",
        kind="fact",
        values=(
            "Austin",
            "Denver",
            "Seattle",
            "Boston",
            "Chicago",
            "Portland",
            "Atlanta",
            "Phoenix",
            "Minneapolis",
            "Pittsburgh",
            "Raleigh",
            "San Diego",
            "Nashville",
            "Columbus",
        ),
        statements=("Lives in {v}.", "User's home is in {v}.", "Based in {v}."),
        updates=("Moved to {v}.", "Just relocated to {v}.", "Now lives in {v} after the move."),
        dev=("Where do I live?", "What city is my home in?"),
        test=(
            "Which city am I based in these days?",
            "Suggest a park near me - what town should you assume I'm in?",
        ),
    ),
    Slot(
        key="employer",
        alt_key="workplace",
        kind="fact",
        values=(
            "Northwind Labs",
            "Helio Health",
            "Brightline Logistics",
            "Cobalt Bank",
            "Fernway Studios",
            "Quarry Analytics",
            "Tidewater Energy",
            "Juniper Schools",
            "Arclight Games",
            "Meridian Insurance",
        ),
        statements=("Works at {v}.", "Employer is {v}.", "Has a job at {v}."),
        updates=(
            "Started a new job at {v}.",
            "Switched companies; now at {v}.",
            "Left for a role at {v}.",
        ),
        dev=("Where do I work?", "What company employs me?"),
        test=("Who pays my salary right now?", "Remind me which firm I'm with."),
    ),
    Slot(
        key="job_title",
        alt_key="role",
        kind="fact",
        values=(
            "data analyst",
            "product manager",
            "nurse practitioner",
            "software engineer",
            "high school teacher",
            "UX researcher",
            "accountant",
            "operations lead",
            "civil engineer",
            "marketing director",
        ),
        statements=("Job title is {v}.", "Works as a {v}.", "Is a {v} by profession."),
        updates=("Promoted; title is now {v}.", "Changed roles to {v}.", "New position: {v}."),
        dev=("What is my job title?", "What do I do for work?"),
        test=("What's my role called at the moment?", "How would you describe my occupation?"),
    ),
    Slot(
        key="diet",
        alt_key="eating_style",
        kind="preference",
        values=(
            "vegetarian",
            "vegan",
            "pescatarian",
            "keto",
            "gluten-free",
            "Mediterranean",
            "dairy-free",
        ),
        statements=("Eats a {v} diet.", "Follows a {v} diet.", "Prefers {v} meals."),
        updates=("Switched to a {v} diet.", "Now eating {v}.", "Changed diet: {v}."),
        dev=("What diet do I follow?", "What kind of food do I eat?"),
        test=(
            "Plan my dinners this week - any rules on what I eat?",
            "Which meals fit my eating habits?",
        ),
    ),
    Slot(
        key="allergy",
        alt_key="allergen",
        kind="constraint",
        values=("peanuts", "shellfish", "penicillin", "tree nuts", "sesame", "latex", "bee stings"),
        statements=(
            "Severely allergic to {v}.",
            "Has a serious {v} allergy.",
            "Must avoid {v}: allergic.",
        ),
        updates=("Allergic to {v}.",),
        dev=("What am I allergic to?", "Do I have any allergies?"),
        test=(
            "Is there anything that could give me a reaction?",
            "What should a restaurant know about my allergy situation?",
        ),
        updatable=False,
    ),
    Slot(
        key="partner_name",
        alt_key="significant_other",
        kind="fact",
        values=(
            "Priya",
            "Marcus",
            "Elena",
            "Jonah",
            "Aisha",
            "Theo",
            "Sofia",
            "Daniel",
            "Mei",
            "Rafael",
        ),
        statements=("Partner's name is {v}.", "Is married to {v}.", "Dating {v}."),
        updates=("Now in a relationship with {v}.", "New partner: {v}.", "Seeing {v} these days."),
        dev=("What is my partner's name?", "Who am I in a relationship with?"),
        test=("Who's my significant other?", "Who should I book the anniversary dinner with?"),
    ),
    Slot(
        key="pet",
        alt_key="animal",
        kind="fact",
        values=(
            "a beagle named Biscuit",
            "a cat named Miso",
            "a corgi named Waffles",
            "a rescue dog named Juniper",
            "two cats, Salt and Pepper",
            "a parrot named Kiwi",
            "a greyhound named Comet",
        ),
        statements=("Has {v}.", "Pet: {v}.", "Lives with {v}."),
        updates=("Adopted {v}.", "Now has {v}.", "Brought home {v}."),
        dev=("What pet do I have?", "What is my dog or cat called?"),
        test=("Who needs feeding when I travel?", "What animal shares my house?"),
    ),
    Slot(
        key="gym_days",
        alt_key="workout_schedule",
        kind="preference",
        values=(
            "Monday and Thursday",
            "Tuesday and Friday",
            "weekends only",
            "Monday, Wednesday and Friday",
            "Tuesday and Saturday",
            "every weekday",
        ),
        statements=("Goes to the gym {v}.", "Works out {v}.", "Gym days: {v}."),
        updates=("Moved gym sessions to {v}.", "Now trains {v}.", "New workout days: {v}."),
        dev=("Which days do I go to the gym?", "When do I work out?"),
        test=(
            "When should you avoid booking things because I'm lifting?",
            "What's my exercise routine?",
        ),
    ),
    Slot(
        key="wake_time",
        alt_key="alarm",
        kind="preference",
        values=("5:30 am", "6:00 am", "6:45 am", "7:15 am", "8:00 am", "4:45 am"),
        statements=("Wakes up at {v}.", "Alarm is set for {v}.", "Up every day at {v}."),
        updates=("Now waking at {v}.", "Changed alarm to {v}.", "Getting up at {v} now."),
        dev=("What time do I wake up?", "When does my alarm go off?"),
        test=("How early do my mornings start?", "What's the earliest I'm usually awake?"),
    ),
    Slot(
        key="favorite_cuisine",
        alt_key="food_favorite",
        kind="preference",
        values=(
            "Thai",
            "Ethiopian",
            "Neapolitan pizza",
            "Korean barbecue",
            "Oaxacan",
            "Sichuan",
            "Lebanese",
            "ramen",
        ),
        statements=("Favorite cuisine is {v}.", "Loves {v} food.", "Top pick for dinner: {v}."),
        updates=("New favorite cuisine: {v}.", "Into {v} lately.", "Now prefers {v}."),
        dev=("What is my favorite cuisine?", "What food do I like most?"),
        test=(
            "Pick a restaurant style I'd enjoy for date night.",
            "What dishes do I get most excited about?",
        ),
    ),
    Slot(
        key="dining_budget",
        alt_key="food_budget",
        kind="constraint",
        values=(
            "$150 a month",
            "$200 a month",
            "$300 a month",
            "$400 a month",
            "$500 a month",
            "$250 a month",
        ),
        statements=(
            "Keeps eating out under {v}.",
            "Dining budget is {v}.",
            "Limit for restaurants: {v}.",
        ),
        updates=(
            "Raised the dining budget to {v}.",
            "Dining budget changed to {v}.",
            "Now caps restaurants at {v}.",
        ),
        dev=("What is my dining budget?", "How much can I spend on restaurants?"),
        test=(
            "How much am I allowed to blow on eating out?",
            "What's my cap for going out to dinner?",
        ),
    ),
    Slot(
        key="doctor",
        alt_key="physician",
        kind="fact",
        values=(
            "Dr. Okafor",
            "Dr. Lindqvist",
            "Dr. Ramirez",
            "Dr. Chen",
            "Dr. Haddad",
            "Dr. Novak",
            "Dr. Patel",
        ),
        statements=("Doctor is {v}.", "Sees {v} for checkups.", "Primary care: {v}."),
        updates=("Switched doctors to {v}.", "New GP: {v}.", "Now sees {v}."),
        dev=("Who is my doctor?", "What's my doctor's name?"),
        test=("Who should I call to book a checkup?", "Which physician looks after me?"),
    ),
    Slot(
        key="car",
        alt_key="vehicle",
        kind="fact",
        values=(
            "a blue Subaru Outback",
            "a white Toyota Prius",
            "a gray Honda Civic",
            "a red Mazda 3",
            "a black Ford F-150",
            "a green Kia Soul",
            "a silver Tesla Model 3",
        ),
        statements=("Drives {v}.", "Car: {v}.", "Owns {v}."),
        updates=("Bought {v}.", "Traded in; now drives {v}.", "Replaced the car with {v}."),
        dev=("What car do I drive?", "What vehicle do I own?"),
        test=("What should I look for in the parking lot?", "What am I driving these days?"),
    ),
    Slot(
        key="sibling",
        alt_key="family_member",
        kind="fact",
        values=(
            "a sister named Mira",
            "a brother named Owen",
            "a sister named Lucia",
            "a brother named Kofi",
            "a sister named Hana",
            "a brother named Idris",
        ),
        statements=("Has {v}.", "Sibling: {v}.", "Close with {v}."),
        updates=("Has {v}.",),
        dev=("Do I have a brother or sister?", "What's my sibling's name?"),
        test=(
            "Who in my family should I call on holidays besides my parents?",
            "Tell me about my siblings.",
        ),
        updatable=False,
    ),
    Slot(
        key="language_goal",
        alt_key="studying",
        kind="goal",
        values=(
            "Japanese",
            "Spanish",
            "Portuguese",
            "Korean",
            "French",
            "German",
            "Italian",
            "Mandarin",
        ),
        statements=("Learning {v}.", "Goal: get conversational in {v}.", "Studying {v} daily."),
        updates=(
            "Dropped the old language; now learning {v}.",
            "Switched to studying {v}.",
            "New language goal: {v}.",
        ),
        dev=("What language am I learning?", "Which language am I studying?"),
        test=("Which vocabulary should my flashcards drill?", "What should my daily lesson be in?"),
    ),
    Slot(
        key="race_goal",
        alt_key="running_target",
        kind="goal",
        values=(
            "a half marathon in October",
            "the Boston Marathon",
            "a spring 10K",
            "a trail ultra in June",
            "a sub-25 minute 5K",
            "the Twin Cities Marathon",
        ),
        statements=("Training for {v}.", "Goal race: {v}.", "Running {v}."),
        updates=(
            "Changed target race to {v}.",
            "Now training for {v}.",
            "Signed up for {v} instead.",
        ),
        dev=("What race am I training for?", "What's my running goal?"),
        test=(
            "What event should my long runs build toward?",
            "Remind me what my training plan is aiming at.",
        ),
    ),
)

# Small talk. "{city}", "{food}" and similar are filled with values from the slot pools,
# usually not the user's own, to create near misses for the retriever.
DISTRACTORS: tuple[str, ...] = (
    "Asked for restaurant ideas in {city} for a friend's visit.",
    "Wanted a weekend itinerary for {city}.",
    "Asked what the weather will be like in {city} next week.",
    "Asked for a {food} recipe to try on Sunday.",
    "Read an article about {food} street food and wanted a summary.",
    "Asked how to write a cover letter for a job at {company}.",
    "Wanted notes on {company}'s quarterly results.",
    "Asked which running shoes are good for beginners.",
    "Asked for a stretching routine after a long run.",
    "Wanted a podcast recommendation about history.",
    "Asked to summarize a book on habits.",
    "Planned a coffee catch-up with an old colleague.",
    "Asked for tips on keeping houseplants alive.",
    "Asked how to dispute a phone bill.",
    "Wanted gift ideas for a coworker's birthday.",
    "Asked how long to boil an egg.",
    "Asked whether {pet_kind} can eat grapes.",
    "Asked for a {language} phrase to greet a neighbor.",
    "Watched a movie about a marathon runner and wanted similar films.",
    "Asked about symptoms of a cold versus the flu.",
    "Asked for a budget template for a vacation.",
    "Wanted to compare two phone plans.",
    "Asked for a packing list for a camping trip.",
    "Asked how to clean a car's interior.",
    "Asked for a morning routine idea from a productivity blog.",
    "Wanted a playlist for the gym.",
    "Asked about visa rules for a trip abroad.",
    "Asked to proofread an email to a landlord.",
    "Wanted to know the best time to book flights.",
    "Asked for icebreaker questions for a team meeting.",
    "Asked what a friend with a {allergen} allergy can eat at a party.",
    "Discussed a colleague who just moved to {city}.",
    "Helped a friend choose between {car} and a minivan.",
)

_FILLERS = {
    "city": SLOTS[0].values,
    "food": ("Thai", "Ethiopian", "Sichuan", "Lebanese", "ramen", "Oaxacan", "Korean"),
    "company": SLOTS[1].values,
    "pet_kind": ("dogs", "cats", "parrots"),
    "language": SLOTS[14].values,
    "allergen": SLOTS[4].values,
    "car": ("a Honda Civic", "a Subaru Outback", "a Tesla Model 3", "a Mazda 3"),
}


@dataclass(frozen=True)
class Probe:
    slot: str
    question: str
    current_id: int  # memory holding the value the user has now
    stale_ids: tuple[int, ...]  # memories holding earlier values
    expected: str
    updated: bool
    stale_values: tuple[str, ...] = ()


@dataclass
class Persona:
    persona_id: int
    store: MemoryStore
    now: datetime
    probes: dict[str, list[Probe]] = field(default_factory=dict)

    @property
    def memories(self) -> list[Memory]:
        return self.store.all(include_inactive=True)


@dataclass(frozen=True)
class _Write:
    when: datetime
    text: str
    kind: str
    key: str | None
    slot: str | None = None
    value: str | None = None
    noise_draw: float = 1.0  # compared with the key-noise level; 1.0 never renamed


def _plan_persona(rng: random.Random, start: datetime, config: dict) -> tuple[list[_Write], dict]:
    sessions = config["sessions_per_persona"]
    gap = timedelta(days=config["days_between_sessions"])
    session_start = [start + i * gap for i in range(sessions)]
    writes: list[_Write] = []
    truth: dict[str, list[str]] = {}

    for slot in SLOTS:
        values = rng.sample(slot.values, k=min(3, len(slot.values)))
        intro = rng.randrange(0, sessions // 2)
        timeline = [(intro, values[0], rng.choice(slot.statements))]
        if slot.updatable and intro < sessions - 1 and rng.random() < config["update_probability"]:
            first = rng.randrange(intro + 1, sessions)
            timeline.append((first, values[1], rng.choice(slot.updates)))
            second_ok = first < sessions - 1 and len(values) > 2
            if second_ok and rng.random() < config["second_update_probability"]:
                second = rng.randrange(first + 1, sessions)
                timeline.append((second, values[2], rng.choice(slot.updates)))
        truth[slot.key] = [v for _, v, _ in timeline]
        for position, (session, value, template) in enumerate(timeline):
            writes.append(
                _Write(
                    when=session_start[session] + timedelta(minutes=rng.randrange(5, 55)),
                    text=template.format(v=value),
                    kind=slot.kind,
                    key=slot.key,
                    slot=slot.key,
                    value=value,
                    noise_draw=rng.random() if position > 0 else 1.0,
                )
            )

    low, high = config["distractors_per_session"]
    for when in session_start:
        for _ in range(rng.randint(low, high)):
            template = rng.choice(DISTRACTORS)
            fills = {name: rng.choice(pool) for name, pool in _FILLERS.items()}
            writes.append(
                _Write(
                    when=when + timedelta(minutes=rng.randrange(5, 55)),
                    text=template.format(**fills),
                    kind="episode",
                    key=None,
                )
            )
    writes.sort(key=lambda w: w.when)
    return writes, truth


def build_personas(config: dict, key_noise: float = 0.0) -> list[Persona]:
    """Generate the benchmark population.

    The same seed gives the same histories at every key-noise level; noise only renames
    the key on some updates, so arms and noise levels are compared on identical data.
    """
    rng = random.Random(config["seed"])
    personas = []
    for persona_id in range(config["personas"]):
        persona_rng = random.Random(rng.getrandbits(64))
        start = datetime(2026, 1, 5, 9, 0) + timedelta(days=persona_rng.randrange(0, 120))
        writes, _truth = _plan_persona(persona_rng, start, config)
        slot_by_key = {s.key: s for s in SLOTS}

        clock = FixedClock(start)
        store = MemoryStore(connect(":memory:"), clock)
        written: dict[str, list[int]] = {}
        values: dict[str, list[str]] = {}
        for write in writes:
            clock.set(write.when)
            key = write.key
            if write.slot is not None and write.noise_draw < key_noise:
                key = slot_by_key[write.slot].alt_key
            memory, _ = store.add(write.text, kind=write.kind, key=key)
            if write.slot is not None and write.value is not None:
                written.setdefault(write.slot, []).append(memory.id)
                values.setdefault(write.slot, []).append(write.value)

        now = max(w.when for w in writes) + timedelta(days=config["days_between_sessions"])
        persona = Persona(persona_id=persona_id, store=store, now=now)
        for split in ("dev", "test"):
            probes = []
            for slot in SLOTS:
                ids = written[slot.key]
                probes.append(
                    Probe(
                        slot=slot.key,
                        question=persona_rng.choice(getattr(slot, split)),
                        current_id=ids[-1],
                        stale_ids=tuple(ids[:-1]),
                        expected=values[slot.key][-1],
                        updated=len(ids) > 1,
                        stale_values=tuple(values[slot.key][:-1]),
                    )
                )
            persona.probes[split] = probes
        personas.append(persona)
    return personas
