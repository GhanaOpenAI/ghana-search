"""Natural Language Processing module for Ghana-MCP.
Uses spaCy (en_core_web_sm) to extract and clean noun chunks (stripping articles,
adverbials, pronouns, and extraneous adjectives), while handling both singular
and plural variants for high-recall search matching.
"""
import re
from typing import List, Dict, Any, Set

try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


FOREIGN_COUNTRIES = {
    "kenya", "nigeria", "uganda", "tanzania", "south africa", "zimbabwe",
    "zambia", "rwanda", "cameroon", "liberia", "sierra leone", "senegal",
    "ivory coast", "côte d'ivoire", "togo", "benin", "mali", "niger",
    "burkina faso", "ethiopia", "somalia", "sudan", "egypt", "angola",
    "mozambique", "botswana", "namibia", "malawi", "gambia", "congo",
    "greece", "russia", "ukraine", "israel", "gaza", "palestine",
    "iran", "iraq", "afghanistan", "pakistan", "india", "china",
    "japan", "germany", "france", "britain"
}

GHANA_CITIES = {
    "accra", "kumasi", "tamale", "takoradi", "tema", "cape coast",
    "sunyani", "koforidua", "ho", "wa", "bolgatanga", "techiman",
    "sekondi", "adum", "lapaz", "madina", "spintex"
}


def is_foreign_title(title: str) -> bool:
    """Return True if title refers to a foreign country and lacks Ghana/Ghanaian context."""
    tl = (title or "").lower()
    if "ghana" in tl or "ghanaian" in tl:
        return False
    for c in FOREIGN_COUNTRIES:
        if re.search(rf"\b{re.escape(c)}\b", tl):
            return True
    return False


def has_word(word: str, text: str) -> bool:
    """True if word appears in text as a whole word (so 'car' doesn't match 'eyecare')."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def token_matches_title(token: str, title: str) -> bool:
    """Match token in title considering singular/plural and morphological derivations."""
    t = token.lower().strip()
    tl = (title or "").lower()
    if not t or len(t) < 2:
        return False
    if has_word(t, tl):
        return True

    # 1. -ies <-> -y (e.g. pharmacies <-> pharmacy, companies <-> company)
    if t.endswith("ies") and len(t) > 4:
        sing_y = t[:-3] + "y"
        if has_word(sing_y, tl):
            return True
    elif t.endswith("y") and len(t) > 2:
        plur_ies = t[:-1] + "ies"
        if has_word(plur_ies, tl):
            return True

    # 2. -es (e.g. branches <-> branch, taxes <-> tax, speeches <-> speech)
    if t.endswith("es") and len(t) > 4:
        sing_es = t[:-2]
        if has_word(sing_es, tl):
            return True
    elif t.endswith(("ch", "sh", "x", "s", "z")):
        plur_es = t + "es"
        if has_word(plur_es, tl):
            return True

    # 3. -s (e.g. bills <-> bill, laws <-> law, books <-> book)
    if t.endswith("s") and len(t) > 3 and not t.endswith("ss"):
        sing_s = t[:-1]
        if has_word(sing_s, tl):
            return True
    if not t.endswith("s"):
        plur_s = t + "s"
        if has_word(plur_s, tl):
            return True

    # 4. Adjective/noun derivations (e.g. parliamentary -> parliament, ghanaian -> ghana)
    if t.endswith("ary") and len(t) > 6 and has_word(t[:-3], tl):
        return True
    if has_word(t + "ary", tl):
        return True
    if t.endswith("ian") and len(t) > 5 and has_word(t[:-3], tl):
        return True
    if has_word(t + "ian", tl):
        return True

    return False


def get_singular_plural_variants(word: str) -> Set[str]:
    """Return both singular and plural forms of a given word."""
    w = word.strip()
    if len(w) < 2:
        return {w}

    variants = {w, w.lower()}
    lower = w.lower()

    # Plural -> Singular
    if lower.endswith("ies") and len(lower) > 4:
        variants.add(w[:-3] + "y")
    elif lower.endswith(("ses", "xes", "zes", "ches", "shes")) and len(lower) > 4:
        variants.add(w[:-2])
    elif lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        variants.add(w[:-1])

    # Singular -> Plural
    if not lower.endswith("s") or lower.endswith("ss"):
        if lower.endswith("y") and not (len(lower) > 2 and lower[-2:] in ("ay", "ey", "oy", "uy")):
            variants.add(w[:-1] + "ies")
        elif lower.endswith(("s", "x", "z", "ch", "sh")):
            variants.add(w + "es")
        else:
            variants.add(w + "s")

    return variants


def _clean_chunk_tokens(chunk) -> List[Any]:
    """Clean a noun chunk by only stripping determiners, articles, pronouns, and punctuation,
    while retaining critical adjectives (e.g. former, current, renewable, chief), nouns, and proper nouns.
    """
    cleaned = []
    for t in chunk:
        # Only strip determiners/articles, adverbs, pronouns, possessive 's, coordinating conjunctions, and punctuation
        if t.pos_ in ("DET", "ADV", "PRON", "PART", "CCONJ", "PUNCT"):
            continue

        # Retain adjectives (e.g. former, current, renewable), nouns, proper nouns, numerals
        cleaned.append(t)

    return cleaned


AUX_VERBS = {
    "be", "is", "was", "are", "were", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "get", "got", "getting", "tell", "show", "find", "give", "know", "see",
    "come", "go", "make", "say", "take"
}


def extract_search_terms(text: str) -> Dict[str, Any]:
    """Extract clean noun chunks, individual nouns, and plural/singular expanded queries."""
    if not text or not text.strip():
        return {
            "clean_chunks": [],
            "nouns": [],
            "search_queries": [],
            "expanded_terms": [],
            "primary_query": "",
        }

    clean_text = text.strip()

    clean_chunks: List[str] = []
    lemma_chunks: List[str] = []
    nouns: List[str] = []
    all_expanded_variants: Set[str] = set()

    if nlp is not None:
        doc = nlp(clean_text)

        # 1. Process and clean each noun chunk
        for chunk in doc.noun_chunks:
            tokens = _clean_chunk_tokens(chunk)
            if not tokens:
                continue

            surface_chunk = " ".join([t.text.strip() for t in tokens if len(t.text.strip()) > 1])
            lemma_chunk = " ".join([t.lemma_.strip() for t in tokens if len(t.lemma_.strip()) > 1])

            if surface_chunk and surface_chunk.lower() not in [c.lower() for c in clean_chunks]:
                clean_chunks.append(surface_chunk)
                all_expanded_variants.add(surface_chunk)

            if lemma_chunk and lemma_chunk.lower() not in [c.lower() for c in lemma_chunks]:
                lemma_chunks.append(lemma_chunk)
                all_expanded_variants.add(lemma_chunk)

            for t in tokens:
                nouns.append(t.text.strip())
                all_expanded_variants.update(get_singular_plural_variants(t.text))
                all_expanded_variants.update(get_singular_plural_variants(t.lemma_))

        # 2. Also capture any standalone nouns, proper nouns, adjectives, or content verbs/gerunds
        for t in doc:
            txt = t.text.strip()
            lemma = t.lemma_.lower().strip()
            if t.is_stop or len(txt) <= 1:
                continue
            if t.pos_ in ("NOUN", "PROPN", "ADJ") or (t.pos_ == "VERB" and lemma not in AUX_VERBS):
                nouns.append(txt)
                all_expanded_variants.update(get_singular_plural_variants(txt))
                all_expanded_variants.update(get_singular_plural_variants(t.lemma_))

    else:
        # Fallback if spaCy is unavailable
        words = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", clean_text)
        stopwords = {
            "what", "who", "how", "where", "when", "why", "which", "can",
            "you", "tell", "about", "the", "and", "for", "with", "from",
            "this", "that", "these", "those", "have", "has", "had", "are",
            "was", "were", "been", "being", "find", "give", "show", "get",
            "very", "too", "so", "a", "an"
        }
        filtered = [w for w in words if w.lower() not in stopwords]
        clean_chunks = [" ".join(filtered)] if filtered else []
        nouns = filtered
        for w in filtered:
            all_expanded_variants.update(get_singular_plural_variants(w))

    # Deduplicate nouns preserving order
    seen = set()
    uniq_nouns = [n for n in nouns if not (n.lower() in seen or seen.add(n.lower()))]

    # If noun chunks missed key terms captured in uniq_nouns, build a comprehensive clean_chunk
    if len(clean_chunks) < len(uniq_nouns) and len(uniq_nouns) > 1:
        combined_phrase = " ".join(uniq_nouns)
        if combined_phrase.lower() not in [c.lower() for c in clean_chunks]:
            clean_chunks.insert(0, combined_phrase)

    # Primary query prioritizes clean noun chunks
    if clean_chunks:
        primary_query = " ".join(clean_chunks)
    elif uniq_nouns:
        primary_query = " ".join(uniq_nouns)
    else:
        primary_query = clean_text

    # Build search query variations (including lemma and plural/singular forms)
    search_queries: List[str] = []
    if primary_query and primary_query not in search_queries:
        search_queries.append(primary_query)

    for c in clean_chunks:
        if c and c not in search_queries:
            search_queries.append(c)

    for lc in lemma_chunks:
        if lc and lc not in search_queries:
            search_queries.append(lc)

    return {
        "clean_chunks": clean_chunks,
        "lemma_chunks": lemma_chunks,
        "nouns": uniq_nouns,
        "expanded_terms": sorted(list(all_expanded_variants)),
        "search_queries": search_queries,
        "primary_query": primary_query,
    }
