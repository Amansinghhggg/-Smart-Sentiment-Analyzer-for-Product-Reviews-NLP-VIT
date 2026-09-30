"""Generate an expanded, realistic sentiment dataset covering diverse categories and common vocabulary."""
import random
import pandas as pd

DOMAINS = [
    # Electronics & Gadgets
    "phone", "laptop", "headphones", "charger", "battery", "camera", "smartwatch", "speaker", "keyboard", "monitor",
    # Food & Dining
    "food", "pizza", "burger", "coffee", "restaurant", "meal", "breakfast", "service",
    # Entertainment & Media
    "movie", "book", "game", "film", "album", "show", "story",
    # Everyday Items
    "backpack", "shoes", "jacket", "watch", "chair", "desk", "bottle", "app", "software"
]

POS_TEMPLATES = [
    "absolutely love this {p}, highly recommend it",
    "works perfectly and the quality is great",
    "excellent {p}, totally worth every penny",
    "very happy and pleased with my purchase",
    "fantastic battery life, beautiful design and sturdy build",
    "amazing value, superb quality and super fast delivery",
    "this {p} exceeded all my expectations, truly wonderful",
    "one of the best products I have ever bought",
    "outstanding performance, extremely smooth and reliable",
    "very easy to use, intuitive and incredibly helpful",
    "top notch quality and exceptional customer service",
    "brilliant design, works like a charm",
    "simply awesome, great experience overall",
    "delicious and fresh, best in town",
    "loved everything about it, solid five stars",
    "impressive speed and crystal clear sound",
    "very durable and feels premium in hand",
    "completely satisfied, great value for money",
    "good quality, works well and arrived quickly",
    "nice {p}, very comfortable and lightweight",
    "great product, no complaints whatsoever",
    "highly pleased, definitely buying again",
    "looks very pretty, charming and elegant",
    "she is very pretty and extremely kind",
    "maam is pretty, smart and helpful",
    "such a pretty, lovely and wonderful person",
]

POS_SHORT = [
    "good", "very good", "great", "excellent", "awesome", "amazing", "superb",
    "loved it", "love this", "fantastic", "wonderful", "perfect", "nice product",
    "highly recommend", "best purchase", "worth it", "five stars", "top quality",
    "pretty", "very pretty", "so pretty", "beautiful", "gorgeous", "cute", "sweet",
    "kind", "lovely", "charming", "handsome", "smart", "polite", "friendly", "attractive"
]

NEG_TEMPLATES = [
    "terrible {p}, stopped working after just two days",
    "complete waste of money, very disappointed",
    "poor quality, cheaply made and not worth the price",
    "the {p} arrived broken and defective, total disaster",
    "awful customer service and extremely slow delivery",
    "do not buy this, it does not work at all",
    "cheap flimsy plastic, fell apart immediately",
    "battery drains within an hour, really frustrating and annoying",
    "horrible experience, regret buying this junk",
    "worst {p} I have ever owned, total trash",
    "useless piece of garbage, request a refund",
    "very poor build, feels like a scam",
    "constantly crashes and full of bugs, unbearable",
    "painfully slow, sluggish and completely unresponsive",
    "tastes awful, cold and stale, never ordering again",
    "bad design, extremely uncomfortable and noisy",
    "fails to perform its basic function, very angry",
    "broken on arrival, return process is a nightmare",
    "bad quality, stopped charging after a week",
    "hate it, completely useless and misleading description",
    "defective unit, zero stars if I could",
    "terrible quality, avoid at all costs"
]

NEG_SHORT = [
    "bad", "very bad", "terrible", "awful", "horrible", "worst", "hate it",
    "waste of money", "broken", "defective", "cheap junk", "trash", "useless",
    "do not buy", "scam", "disappointed", "poor quality", "one star",
    "ugly", "rude", "mean", "annoying", "stupid", "foolish"
]

NEU_TEMPLATES = [
    "the {p} is okay, nothing special about it",
    "average {p}, does the job as expected",
    "it is fine for the price, neither good nor bad",
    "arrived on time, works exactly as described",
    "decent {p} but nothing impressive to write home about",
    "mixed feelings, some parts are good while others are not",
    "not bad, not great either, strictly so-so",
    "the {p} is acceptable for basic everyday use",
    "fair quality for the price, standard features",
    "it works fine, ordinary build and expected performance",
    "tolerable experience, meets expectations but does not stand out",
    "middle of the road, reasonable for what you pay",
    "functions as advertised, nothing more nothing less",
    "moderate quality, adequate for short-term use",
    "it is alright, quite standard and unremarkable",
    "average performance, neither impressed nor disappointed"
]

NEU_SHORT = [
    "okay", "ok", "average", "fine", "decent", "not bad", "so so",
    "fair", "normal", "acceptable", "just okay", "as expected", "mediocre"
]


def make_dataset(n_per_class: int = 650, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    categories = [
        ("Positive", POS_TEMPLATES, POS_SHORT, (4, 5)),
        ("Negative", NEG_TEMPLATES, NEG_SHORT, (1, 2)),
        ("Neutral", NEU_TEMPLATES, NEU_SHORT, (3,)),
    ]

    for label, templates, shorts, ratings in categories:
        for s in shorts:
            for _ in range(8):
                rows.append({"review": s, "rating": rng.choice(ratings), "label": label})
                rows.append({"review": f"it is {s}.", "rating": rng.choice(ratings), "label": label})
                rows.append({"review": f"she is {s}.", "rating": rng.choice(ratings), "label": label})

        remaining = n_per_class - len([r for r in rows if r["label"] == label])
        for _ in range(max(100, remaining)):
            p = rng.choice(DOMAINS)
            num_phrases = rng.choice([1, 2])
            selected = [rng.choice(templates).format(p=p) for _ in range(num_phrases)]
            text = ". ".join(selected).capitalize() + "."
            rows.append({"review": text, "rating": rng.choice(ratings), "label": label})

    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("data/sample_reviews.csv", index=False)
    print(f"Wrote data/sample_reviews.csv with {len(df)} reviews across classes:")
    print(df["label"].value_counts())
