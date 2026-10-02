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

# Second held-out set, written after the first test run and before any embedding
# experiment (docs/ANALYSIS_PLAN.md, "Round 2"). Each detail gets one direct wording that
# names its topic and one indirect wording that only implies it.
HOLDOUT: dict[str, tuple[str, str]] = {
    "home_city": (
        "What city do I call home?",
        "Which local news station should I watch for my weather?",
    ),
    "employer": ("Which company do I work for now?", "Whose logo is on my work badge?"),
    "job_title": (
        "What is my current job?",
        "What do I tell people I do when they ask at parties?",
    ),
    "diet": ("Do I follow any particular diet?", "Can you pick a sandwich for me at the deli?"),
    "allergy": (
        "Which allergies do I have?",
        "Is there anything I should tell the waiter before ordering?",
    ),
    "partner_name": (
        "What's the name of my partner?",
        "Who am I sharing the hotel room with on our getaway?",
    ),
    "pet": ("What kind of pet do I own?", "Who is the vet appointment for?"),
    "gym_days": (
        "What days are my workouts?",
        "Which evenings should you keep clear so I can get my reps in?",
    ),
    "wake_time": ("When do I usually get up?", "Is a 6 am call too early for me?"),
    "favorite_cuisine": (
        "What's my favorite type of food?",
        "What should we order in tonight to cheer me up?",
    ),
    "dining_budget": (
        "What's my monthly restaurant budget?",
        "Can I afford another fancy dinner out this month?",
    ),
    "doctor": (
        "What's the name of my doctor?",
        "Who should the pharmacy send my prescription questions to?",
    ),
    "car": ("What car do I own?", "What model should I tell the mechanic I'm bringing in?"),
    "sibling": ("Do I have any siblings?", "Who else grew up in the same house as me?"),
    "language_goal": (
        "Which language am I learning at the moment?",
        "Which country's movies should I watch with subtitles for practice?",
    ),
    "race_goal": (
        "Which race am I training for?",
        "What finish line am I working toward this season?",
    ),
}

# Third held-out set, for round 4 (transformer sentence encoders). Written after round 2
# was finished and before any transformer model was run on benchmark text.
HOLDOUT2: dict[str, tuple[str, str]] = {
    "home_city": (
        "Which city do I live in?",
        "If I say 'let's grab brunch downtown', which downtown do I mean?",
    ),
    "employer": (
        "What's the name of the company I work at?",
        "Whose quarterly all-hands meeting do I attend?",
    ),
    "job_title": ("What's my job title?", "How would a coworker introduce me to a new client?"),
    "diet": ("What diet am I on?", "Which of the potluck dishes will I actually want to try?"),
    "allergy": (
        "Tell me my allergies.",
        "What should be on the medical alert bracelet I'm ordering?",
    ),
    "partner_name": ("Who is my partner?", "Whose name goes next to mine on the joint lease?"),
    "pet": ("Tell me about my pet.", "Who will the house sitter be looking after?"),
    "gym_days": (
        "On which days do I go to the gym?",
        "Which days should my spotter expect me?",
    ),
    "wake_time": (
        "What time does my alarm ring?",
        "Can we schedule a sunrise hike without me oversleeping?",
    ),
    "favorite_cuisine": (
        "Which cuisine do I love most?",
        "What kind of cookbook would make a good gift for me?",
    ),
    "dining_budget": (
        "How big is my dining-out budget?",
        "Should I say yes to a pricey tasting menu this month?",
    ),
    "doctor": ("Who's my GP?", "Whose office should get my updated insurance card?"),
    "car": ("Which car is mine?", "What should I tell the valet to bring around?"),
    "sibling": (
        "What's my brother's or sister's name?",
        "Who could be my kids' aunt or uncle on my side?",
    ),
    "language_goal": (
        "Which foreign language am I working on?",
        "Which app course should I open for my daily streak?",
    ),
    "race_goal": (
        "What race is my goal?",
        "Which bib number pickup should I put in my calendar?",
    ),
}

# Fourth held-out set, for round 5 (a wider weight grid for the transformer encoder).
# Written after round 4 was finished and before any round 5 run.
HOLDOUT3: dict[str, tuple[str, str]] = {
    "home_city": (
        "Where am I living these days?",
        "Which city's transit card should I keep topped up?",
    ),
    "employer": (
        "Which business am I employed by?",
        "Whose company holiday party am I going to?",
    ),
    "job_title": (
        "What title is on my business card?",
        "What do I say when a stranger asks how I spend my workday?",
    ),
    "diet": ("Describe my eating habits.", "Which dishes should the caterer leave off my plate?"),
    "allergy": (
        "Do I have any allergies I should mention?",
        "What should the ER nurse know before giving me anything?",
    ),
    "partner_name": ("Who am I dating or married to?", "Who's my plus-one for the wedding?"),
    "pet": ("What animal do I have at home?", "Who gets the leftover chicken scraps at my place?"),
    "gym_days": ("Which days do I work out?", "When do I pack my lifting shoes?"),
    "wake_time": (
        "When do I wake up in the morning?",
        "Is a 7 am breakfast meeting realistic for me?",
    ),
    "favorite_cuisine": (
        "What's my go-to cuisine?",
        "Which takeout menu should stay on my fridge?",
    ),
    "dining_budget": (
        "What's my monthly limit for eating out?",
        "Can I afford splitting a steak dinner with friends this week?",
    ),
    "doctor": (
        "Who is the doctor I see?",
        "Who should I ask for a referral to a specialist?",
    ),
    "car": ("What do I drive?", "What's parked in my driveway?"),
    "sibling": (
        "Do I have a sister or brother?",
        "Who will my parents' other child be at the reunion?",
    ),
    "language_goal": (
        "What language am I trying to learn?",
        "Which country's newspapers should I read for practice?",
    ),
    "race_goal": (
        "Which race have I signed up for?",
        "What's the big event my weekly mileage is building to?",
    ),
}

# Fifth held-out set, for round 6 (larger transformer encoders). Written after round 5 was
# merged and before any larger encoder was run on benchmark text.
HOLDOUT4: dict[str, tuple[str, str]] = {
    "home_city": (
        "Which city is my home base?",
        "Whose local sports team should I cheer for?",
    ),
    "employer": (
        "Who is my current employer?",
        "Whose name is on the logo of my work laptop?",
    ),
    "job_title": (
        "What is my job called?",
        "What would I put under 'current position' on a networking site?",
    ),
    "diet": (
        "What kind of diet do I keep?",
        "What should the host know before planning a dinner party menu for me?",
    ),
    "allergy": (
        "What allergies do I have?",
        "What should I warn the flight attendant about before the snack cart comes?",
    ),
    "partner_name": (
        "What's my partner called?",
        "Who should be listed as my emergency contact at home?",
    ),
    "pet": (
        "Do I own a pet?",
        "Who should the dog walker or pet sitter come to see?",
    ),
    "gym_days": (
        "What is my gym schedule?",
        "Which days am I usually sore from training?",
    ),
    "wake_time": (
        "What time am I usually up?",
        "When does my day start?",
    ),
    "favorite_cuisine": (
        "What type of cuisine do I like best?",
        "Where should we book to celebrate my promotion?",
    ),
    "dining_budget": (
        "How much do I budget for restaurants?",
        "How much should I spend on takeout before I'm over for the month?",
    ),
    "doctor": (
        "Which doctor do I see for checkups?",
        "Who should I phone when this cough won't go away?",
    ),
    "car": (
        "Which car do I have?",
        "What's the make and model on my insurance renewal?",
    ),
    "sibling": (
        "Do I have siblings, and who?",
        "Who shares my parents?",
    ),
    "language_goal": (
        "What language am I studying right now?",
        "Which subtitles should I turn on to practice?",
    ),
    "race_goal": (
        "What race am I preparing for?",
        "What event is my long Sunday run for?",
    ),
}

# Sixth held-out set, for round 7 (cross-encoder reranking). Written after round 6 was
# merged and before any cross-encoder was run on benchmark text.
HOLDOUT5: dict[str, tuple[str, str]] = {
    "home_city": (
        "Which city do I live in now?",
        "Which city's weather forecast matters to me most?",
    ),
    "employer": (
        "Which company am I working at?",
        "Whose holiday party am I going to this year?",
    ),
    "job_title": (
        "What's my official job title?",
        "What do I say when someone at a party asks what I do?",
    ),
    "diet": (
        "What are my eating habits?",
        "What should I tick on the wedding RSVP meal card?",
    ),
    "allergy": (
        "Which allergy have I told you about?",
        "What should the waiter check the kitchen for when I order?",
    ),
    "partner_name": (
        "Who is my significant other?",
        "Whose name goes next to mine on the anniversary card?",
    ),
    "pet": (
        "What pet lives with me?",
        "Who will need a new collar and tag?",
    ),
    "gym_days": (
        "When do I go to the gym?",
        "Which evenings am I busy with workouts?",
    ),
    "wake_time": (
        "When do I wake up?",
        "How early should I set my alarm?",
    ),
    "favorite_cuisine": (
        "Which cuisine is my favourite?",
        "Which takeout menu would I reach for first?",
    ),
    "dining_budget": (
        "What's my monthly eating-out budget?",
        "Can I afford another dinner out this month?",
    ),
    "doctor": (
        "Who is my physician?",
        "Whose office should send my blood test results?",
    ),
    "car": (
        "What car am I driving these days?",
        "Which model should I search for when ordering new wiper blades?",
    ),
    "sibling": (
        "Who are my brothers or sisters?",
        "Who else did my parents raise?",
    ),
    "language_goal": (
        "Which language am I learning?",
        "Which language app course should I open tonight?",
    ),
    "race_goal": (
        "Which race is my goal?",
        "What finish line am I working toward?",
    ),
}

# Seventh held-out set, for round 8 (quantized models). Written after round 7 was merged and
# before any quantized model was run on benchmark text.
HOLDOUT6: dict[str, tuple[str, str]] = {
    "home_city": (
        "Where am I living at the moment?",
        "Which city should my mail be forwarded to?",
    ),
    "employer": (
        "Who do I work for these days?",
        "Whose badge do I swipe to get into the office?",
    ),
    "job_title": (
        "What is my role at work?",
        "What title should go on my new business cards?",
    ),
    "diet": (
        "Do I follow a particular diet?",
        "What should I tell the caterer about my meals?",
    ),
    "allergy": (
        "Am I allergic to anything?",
        "What should the school nurse know about my reactions?",
    ),
    "partner_name": (
        "Who is the person I'm dating?",
        "Who should I book the couples massage with?",
    ),
    "pet": (
        "Which pet do I look after?",
        "Who needs feeding while I'm away for the weekend?",
    ),
    "gym_days": (
        "On which days do I hit the gym?",
        "When should I avoid booking evening meetings because of training?",
    ),
    "wake_time": (
        "What time do I normally wake up?",
        "When is it too early to send me a text?",
    ),
    "favorite_cuisine": (
        "Which kind of food is my top pick?",
        "Which restaurant should I pick for my birthday dinner?",
    ),
    "dining_budget": (
        "What's my restaurant spending limit each month?",
        "How much is left for dining out before I go over?",
    ),
    "doctor": (
        "Who is my GP?",
        "Who should I ask for a prescription refill?",
    ),
    "car": (
        "Which vehicle do I own?",
        "What car should the valet expect?",
    ),
    "sibling": (
        "Do I have any brothers or sisters?",
        "Who will I see at the family reunion besides my parents?",
    ),
    "language_goal": (
        "What language do I practise on my lunch break?",
        "Which language exchange meetup should I join?",
    ),
    "race_goal": (
        "Which race am I training for right now?",
        "What's the long-distance event on my calendar?",
    ),
}

# Eighth held-out set, for round 9 (smaller encoders). Written after round 8 was merged and
# before any of the smaller encoders was run on benchmark text in this round.
HOLDOUT7: dict[str, tuple[str, str]] = {
    "home_city": (
        "What's the name of the city I live in?",
        "Which city's council tax am I paying?",
    ),
    "employer": (
        "Which firm currently employs me?",
        "Whose payroll am I on?",
    ),
    "job_title": (
        "What position do I hold at work?",
        "How would my manager describe my job on an org chart?",
    ),
    "diet": (
        "What kind of eating plan do I stick to?",
        "Which meals can I order at the office lunch?",
    ),
    "allergy": (
        "Which allergy do I live with?",
        "What ingredient could send me to the emergency room?",
    ),
    "partner_name": (
        "What's my partner's name?",
        "Whose name should go on the second plane ticket for our holiday?",
    ),
    "pet": (
        "What kind of pet do I have?",
        "Who needs a vet booster this year?",
    ),
    "gym_days": (
        "Which days are my gym days?",
        "When should my trainer expect me this week?",
    ),
    "wake_time": (
        "At what time do I get up?",
        "What's the earliest I could take a call?",
    ),
    "favorite_cuisine": (
        "Which cuisine would I choose above all others?",
        "What kind of restaurant should we book for date night?",
    ),
    "dining_budget": (
        "What's my budget for eating out?",
        "Is a big group dinner this month within what I set aside?",
    ),
    "doctor": (
        "Which doctor looks after my health?",
        "Who should sign my sick note?",
    ),
    "car": (
        "What model of car do I own?",
        "What should I tell the mechanic I'm bringing in?",
    ),
    "sibling": (
        "Who is my sibling?",
        "Who should I call to plan our parents' anniversary surprise?",
    ),
    "language_goal": (
        "What language am I taking lessons in?",
        "Which dictionary app should I keep on my phone?",
    ),
    "race_goal": (
        "What race have I entered?",
        "Which start line am I training to reach?",
    ),
}

# Ninth held-out set, for round 12 (larger encoders). Written after round 11 was merged and
# before any of the larger encoders was run on benchmark text.
HOLDOUT8: dict[str, tuple[str, str]] = {
    "home_city": (
        "Which city do I call home?",
        "Whose local weather forecast should you check for me?",
    ),
    "employer": (
        "Which organisation am I working for?",
        "Whose name is printed on my staff lanyard?",
    ),
    "job_title": (
        "What's my job called?",
        "What should I put under occupation on this form?",
    ),
    "diet": (
        "Which diet am I on?",
        "What should the caterer know before preparing my plate?",
    ),
    "allergy": (
        "What allergy have I got?",
        "Which ingredient do I always check labels for?",
    ),
    "partner_name": (
        "Who am I with romantically?",
        "Who should be my plus-one at the wedding?",
    ),
    "pet": (
        "Which animal do I keep as a pet?",
        "Who will the pet sitter be looking after?",
    ),
    "gym_days": (
        "On which days do I train at the gym?",
        "Which evenings are already taken by my workouts?",
    ),
    "wake_time": (
        "When do I normally wake?",
        "How early can you schedule my morning reminder?",
    ),
    "favorite_cuisine": (
        "What's the cuisine I like best?",
        "What should I cook to treat myself tonight?",
    ),
    "dining_budget": (
        "How much have I budgeted for restaurants?",
        "Can I afford the tasting menu this month?",
    ),
    "doctor": (
        "Who is my family doctor?",
        "Whose surgery should I ring about this rash?",
    ),
    "car": (
        "What car is parked in my driveway?",
        "What make should the insurance quote be for?",
    ),
    "sibling": (
        "What's the name of my brother or sister?",
        "Who grew up in the same house as me?",
    ),
    "language_goal": (
        "Which language am I trying to learn?",
        "What should I set the subtitles to for practice?",
    ),
    "race_goal": (
        "Which race am I preparing for?",
        "What finish line am I aiming to cross?",
    ),
}

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
        # Held-out wordings for rounds 2 and 4 to 12: fixed, so they use no randomness and
        # leave the dev and test probes above exactly as they were.
        for prefix, wordings in (
            ("holdout", HOLDOUT),
            ("holdout2", HOLDOUT2),
            ("holdout3", HOLDOUT3),
            ("holdout4", HOLDOUT4),
            ("holdout5", HOLDOUT5),
            ("holdout6", HOLDOUT6),
            ("holdout7", HOLDOUT7),
            ("holdout8", HOLDOUT8),
        ):
            for style, position in (("direct", 0), ("indirect", 1)):
                persona.probes[f"{prefix}_{style}"] = [
                    Probe(
                        slot=slot.key,
                        question=wordings[slot.key][position],
                        current_id=written[slot.key][-1],
                        stale_ids=tuple(written[slot.key][:-1]),
                        expected=values[slot.key][-1],
                        updated=len(written[slot.key]) > 1,
                        stale_values=tuple(values[slot.key][:-1]),
                    )
                    for slot in SLOTS
                ]
            persona.probes[prefix] = (
                persona.probes[f"{prefix}_direct"] + persona.probes[f"{prefix}_indirect"]
            )
        personas.append(persona)
    return personas
