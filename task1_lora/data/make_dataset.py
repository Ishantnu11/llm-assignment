"""
Builds a synthetic dataset for the task:
    plain-English expense note  ->  fixed JSON schema

Schema (always all 5 keys, in this order):
{"amount": float, "currency": "INR"|"USD"|"EUR", "category": one of CATEGORIES,
 "merchant": str or null, "date": "today"|"yesterday"|"YYYY-MM-DD"}

Run:  python data/make_dataset.py        (writes train.jsonl / val.jsonl / test.jsonl)
The test set uses templates the model never sees in training (held-out phrasing),
so the score measures generalisation, not memorisation.
"""
import json, random
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent

CATEGORIES = {
    "food":          [("Swiggy", "dinner"), ("Zomato", "biryani"), ("Starbucks", "coffee"), ("Dominos", "pizza"), (None, "lunch"), ("McDonald's", "burgers")],
    "travel":        [("Uber", "a cab"), ("Ola", "a ride"), ("IndiGo", "a flight"), ("IRCTC", "train tickets"), (None, "petrol"), ("Rapido", "a bike taxi")],
    "shopping":      [("Amazon", "headphones"), ("Flipkart", "a phone case"), ("Myntra", "a jacket"), ("Decathlon", "running shoes"), (None, "a gift")],
    "bills":         [("Airtel", "the phone bill"), ("Jio", "recharge"), ("BSES", "electricity"), (None, "rent"), ("ACT", "wifi")],
    "entertainment": [("Netflix", "subscription"), ("PVR", "movie tickets"), ("Spotify", "premium"), ("BookMyShow", "concert tickets"), ("Steam", "a game")],
    "health":        [("Apollo", "medicines"), ("Cult.fit", "gym membership"), (None, "a doctor visit"), ("1mg", "vitamins"), ("PharmEasy", "a prescription")],
}
CURRENCY_FORMS = {
    "INR": ["₹{a}", "Rs {a}", "{a} rupees", "INR {a}", "{a} rs"],
    "USD": ["${a}", "{a} dollars", "USD {a}", "{a} bucks"],
    "EUR": ["€{a}", "{a} euros", "EUR {a}"],
}
DATES = [("today", "today"), ("yesterday", "yesterday"), ("", "today")]  # empty phrase -> default today
for _ in range(10):
    m, d = random.randint(1, 12), random.randint(1, 28)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    DATES.append((f"on {d} {months[m-1]} 2026", f"2026-{m:02d}-{d:02d}"))
    DATES.append((f"on 2026-{m:02d}-{d:02d}", f"2026-{m:02d}-{d:02d}"))

TRAIN_TEMPLATES = [
    "spent {money} on {item} at {merchant} {date}",
    "paid {money} to {merchant} for {item} {date}",
    "{date} I spent {money} on {item} from {merchant}",
    "log {money} for {item}, {merchant}, {date}",
    "add expense: {item} {money} {merchant} {date}",
    "{merchant} charged me {money} for {item} {date}",
]
TRAIN_TEMPLATES_NOMERCH = [
    "spent {money} on {item} {date}",
    "paid {money} for {item} {date}",
    "add expense {item} {money} {date}",
]
# Held-out phrasings used ONLY in the test set
TEST_TEMPLATES = [
    "ugh, {merchant} just took {money} for {item} {date}",
    "note to self: {item} cost {money} ({merchant}) {date}",
    "{money} gone on {item} via {merchant} {date} lol",
    "can you track {item} - {money} - {merchant} - {date}",
]
TEST_TEMPLATES_NOMERCH = [
    "{item} set me back {money} {date}",
    "{date} dropped {money} on {item}, no receipt",
]

SYSTEM = ("Convert the user's expense note into JSON with exactly these keys: "
          "amount (number), currency (INR, USD or EUR), category (food, travel, shopping, bills, entertainment, health), "
          "merchant (string or null), date (today, yesterday or YYYY-MM-DD). Output only the JSON.")


def make_example(templates, templates_nomerch):
    category = random.choice(list(CATEGORIES))
    merchant, item = random.choice(CATEGORIES[category])
    currency = random.choices(["INR", "USD", "EUR"], weights=[6, 3, 1])[0]
    amount = random.choice([random.randint(20, 5000), round(random.uniform(1, 300), 2)])
    money = random.choice(CURRENCY_FORMS[currency]).format(a=amount)
    date_phrase, date_val = random.choice(DATES)
    tpl = random.choice(templates if merchant else templates_nomerch)
    text = tpl.format(money=money, item=item, merchant=merchant, date=date_phrase)
    text = " ".join(text.split()).strip(" ,")
    target = {"amount": float(amount), "currency": currency, "category": category,
              "merchant": merchant, "date": date_val}
    return {"system": SYSTEM, "input": text, "output": json.dumps(target, ensure_ascii=False)}


def dump(name, rows):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    dump("train.jsonl", [make_example(TRAIN_TEMPLATES, TRAIN_TEMPLATES_NOMERCH) for _ in range(1200)])
    dump("val.jsonl",   [make_example(TRAIN_TEMPLATES, TRAIN_TEMPLATES_NOMERCH) for _ in range(100)])
    dump("test.jsonl",  [make_example(TEST_TEMPLATES, TEST_TEMPLATES_NOMERCH) for _ in range(100)])
