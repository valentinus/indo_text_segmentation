"""Registry of Indonesian local-language Wikipedias used for text segmentation.

`wiki` is the subdomain (e.g. https://jv.wikipedia.org) and `dbname` is the
prefix used on dumps.wikimedia.org (hyphens become underscores).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    skip_headings: FrozenSet[str] = field(default_factory=frozenset)

    @property
    def dbname(self) -> str:
        return self.code.replace('-', '_') + 'wiki'

    def dump_url(self, date: str = 'latest') -> str:
        return (f'https://dumps.wikimedia.org/{self.dbname}/{date}/'
                f'{self.dbname}-{date}-pages-articles.xml.bz2')


# Original Indonesian list, kept exactly as before for reproducibility.
_ID_SKIP = {'Catatan', 'Referensi', 'Lihat pula', 'Pranala luar', 'Bahan bacaan terkait'}

# Boilerplate section titles (references, see also, external links, ...) seen
# across Indonesian local-language Wikipedias. Matched case-insensitively.
# Extend with `scripts/corpus_stats.py --top-headings` output if needed.
_COMMON_SKIP = _ID_SKIP | {
    # Indonesian variants often reused by regional wikis
    'Rujukan', 'Catatan kaki', 'Daftar pustaka', 'Bacaan lanjutan', 'Sumber', 'Pustaka',
    'Galeri', 'Referensi lan cathetan',
    # Javanese
    'Cathetan', 'Cathetan suku', 'Pranala njaba', 'Pranala jaba', 'Pranala jawi',
    'Deleng uga', 'Delengen uga', 'Waosan salajengipun', 'Kapustakan', 'Sumber rujukan',
    # Sundanese
    'Tumbu kaluar', 'Tutumbu kaluar', 'Tempo ogé', 'Tempo oge', 'Tingali ogé', 'Tingali oge',
    'Catetan', 'Référénsi', 'Bacaan salajengna',
    # Minangkabau
    'Rujuakan', 'Pautan lua', 'Pranala lua', 'Caliak pulo', 'Caliak juo', 'Bacaan lanjuik',
    # Banjar / Betawi / others
    'Tautan luar', 'Lihat jua', 'Liat juga', 'Tilik jua',
}

LANGUAGES: Dict[str, Language] = {l.code: l for l in [
    Language('id', 'Indonesian', frozenset(_ID_SKIP)),
    Language('jv', 'Javanese', frozenset(_COMMON_SKIP)),
    Language('su', 'Sundanese', frozenset(_COMMON_SKIP)),
    Language('min', 'Minangkabau', frozenset(_COMMON_SKIP)),
    Language('ban', 'Balinese', frozenset(_COMMON_SKIP)),
    Language('bjn', 'Banjar', frozenset(_COMMON_SKIP)),
    Language('ace', 'Acehnese', frozenset(_COMMON_SKIP)),
    Language('bug', 'Buginese', frozenset(_COMMON_SKIP)),
    Language('mad', 'Madurese', frozenset(_COMMON_SKIP)),
    Language('gor', 'Gorontalo', frozenset(_COMMON_SKIP)),
    Language('map-bms', 'Banyumasan', frozenset(_COMMON_SKIP)),
    Language('nia', 'Nias', frozenset(_COMMON_SKIP)),
    Language('bew', 'Betawi', frozenset(_COMMON_SKIP)),
    Language('bbc', 'Batak Toba', frozenset(_COMMON_SKIP)),
    Language('btm', 'Batak Mandailing', frozenset(_COMMON_SKIP)),
]}

LOCAL_LANGUAGES = [c for c in LANGUAGES if c != 'id']


def get_language(code: str) -> Language:
    if code in LANGUAGES:
        return LANGUAGES[code]
    # Unknown code: still usable, with the shared boilerplate list.
    return Language(code, code, frozenset(_COMMON_SKIP))
