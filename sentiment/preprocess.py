"""Text preprocessing: lowercase, HTML/URL removal, tokenization, stopwords, lemmatization."""
import re
from html import unescape

FALLBACK_STOP = set("""a an the and or but if while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when where why
how all any both each few more most other some such only own same so than too very s t can will just i me my myself
we our you your he him his she her it its they them their what which who whom this that these those am is are was were
be been being have has had having do does did doing""".split())
NEGATIONS = {"no", "nor", "not", "never", "neither", "cannot"}

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z]+")
_resources = None


def _load_resources():
    """Use NLTK stopwords/WordNet if available (downloads once); otherwise fall back gracefully."""
    global _resources
    if _resources is not None:
        return _resources
    stop, lemm = set(FALLBACK_STOP), None
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        for pkg, path in (("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")):
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)
        try:
            stop = set(stopwords.words("english"))
        except Exception:
            pass
        try:
            l = WordNetLemmatizer(); l.lemmatize("tests"); lemm = l
        except Exception:
            lemm = None
    except Exception:
        pass
    _resources = (stop - NEGATIONS, lemm)
    return _resources


def clean_text(text: str) -> str:
    """Return a cleaned, space-joined token string. Negations (not, no, never) are kept."""
    stop, lemm = _load_resources()
    text = unescape(str(text)).lower()
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"n't\b", " not", text)
    tokens = [t for t in _TOKEN.findall(text) if t not in stop and len(t) > 1]
    if lemm:
        tokens = [lemm.lemmatize(t) for t in tokens]
    return " ".join(tokens)
