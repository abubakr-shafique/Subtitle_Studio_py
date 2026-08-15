"""Language registry shared by transcription (Whisper) and translation backends."""
from collections import OrderedDict

# display name -> per-backend language codes
LANGUAGES = OrderedDict([
    ("English",               {"whisper": "en", "google": "en",    "nllb": "eng_Latn"}),
    ("Turkish",               {"whisper": "tr", "google": "tr",    "nllb": "tur_Latn"}),
    ("Spanish",               {"whisper": "es", "google": "es",    "nllb": "spa_Latn"}),
    ("Hindi",                 {"whisper": "hi", "google": "hi",    "nllb": "hin_Deva"}),
    ("Urdu",                  {"whisper": "ur", "google": "ur",    "nllb": "urd_Arab"}),
    ("French",                {"whisper": "fr", "google": "fr",    "nllb": "fra_Latn"}),
    ("Chinese (Simplified)",  {"whisper": "zh", "google": "zh-CN", "nllb": "zho_Hans"}),
    ("Chinese (Traditional)", {"whisper": "zh", "google": "zh-TW", "nllb": "zho_Hant"}),
    ("Japanese",              {"whisper": "ja", "google": "ja",    "nllb": "jpn_Jpan"}),
    ("Korean",                {"whisper": "ko", "google": "ko",    "nllb": "kor_Hang"}),
    ("Arabic",                {"whisper": "ar", "google": "ar",    "nllb": "arb_Arab"}),
    ("German",                {"whisper": "de", "google": "de",    "nllb": "deu_Latn"}),
    ("Russian",               {"whisper": "ru", "google": "ru",    "nllb": "rus_Cyrl"}),
    ("Portuguese",            {"whisper": "pt", "google": "pt",    "nllb": "por_Latn"}),
    ("Italian",               {"whisper": "it", "google": "it",    "nllb": "ita_Latn"}),
    ("Dutch",                 {"whisper": "nl", "google": "nl",    "nllb": "nld_Latn"}),
    ("Bengali",               {"whisper": "bn", "google": "bn",    "nllb": "ben_Beng"}),
    ("Punjabi",               {"whisper": "pa", "google": "pa",    "nllb": "pan_Guru"}),
    ("Persian",               {"whisper": "fa", "google": "fa",    "nllb": "pes_Arab"}),
    ("Vietnamese",            {"whisper": "vi", "google": "vi",    "nllb": "vie_Latn"}),
    ("Thai",                  {"whisper": "th", "google": "th",    "nllb": "tha_Thai"}),
    ("Indonesian",            {"whisper": "id", "google": "id",    "nllb": "ind_Latn"}),
    ("Polish",                {"whisper": "pl", "google": "pl",    "nllb": "pol_Latn"}),
    ("Ukrainian",             {"whisper": "uk", "google": "uk",    "nllb": "ukr_Cyrl"}),
])


def _code(name, backend):
    try:
        return LANGUAGES[name][backend]
    except KeyError:
        raise ValueError(f"Unsupported language: {name!r}") from None


def whisper_code(name):
    return _code(name, "whisper")


def google_code(name):
    return _code(name, "google")


def nllb_code(name):
    return _code(name, "nllb")


def name_for_whisper(code):
    """Display name for a Whisper ISO code, or None if it is not in the registry."""
    for name, codes in LANGUAGES.items():
        if codes["whisper"] == code:
            return name
    return None


def nllb_for_whisper(code):
    name = name_for_whisper(code)
    if name is None:
        raise ValueError(
            f"Whisper language code {code!r} is not in the NLLB mapping - "
            "pick the source language manually."
        )
    return LANGUAGES[name]["nllb"]
