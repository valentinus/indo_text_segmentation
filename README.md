# Indonesian & Local-Language Text Segmentation

Evaluates text segmentation methods (TextTiling, C99, CVS, TopicTiling, BayesSeg,
BayesSeg-MD, BeamSeg, BERT, BERTopic) on Wikipedia articles. Gold boundaries are
section headings (`<h2>`/`<h3>`). Scores are Pk and WindowDiff (lower is better).

Supported Wikipedias (`languages.py`): `id` Indonesian, `jv` Javanese, `su` Sundanese,
`min` Minangkabau, `ban` Balinese, `bjn` Banjar, `ace` Acehnese, `bug` Buginese,
`mad` Madurese, `gor` Gorontalo, `map-bms` Banyumasan, `nia` Nias, `bew` Betawi,
`bbc` Batak Toba, `btm` Batak Mandailing. Other wiki codes work too.

```bash
pip install -r requirements.txt
```

## 1. Prepare a corpus

The Indonesian corpus is in `text/`. Other languages go to `data/wiki/<lang>/text/`.

```bash
# from a dump you already have
python scripts/prepare_wiki.py --lang jv --dump /path/to/jvwiki-latest-pages-articles.xml.bz2

# or download from dumps.wikimedia.org and extract
python scripts/prepare_wiki.py --lang su min ban
python scripts/prepare_wiki.py --lang all
```

This runs `wikiextractor --html` through a patched copy of the installed package, with two fixes:

* wikiextractor <= 3.1.0 crashes with `IndexError` on some nested lists.
* wikiextractor 3.1.0 makes `--html` imply `--links`. The patch turns that off so the
  output has the same format as `text/`.

Raw `.bz2` dumps are git-ignored. The extracted `text/` folders can be committed.

## 2. Inspect the corpus

```bash
python scripts/corpus_stats.py --lang id jv --top-headings 40
```

This prints the number of usable documents and the most frequent headings. Headings
marked `[skip]` are boilerplate (references, see also, external links) that are removed
before segmentation. Add missing ones to `languages.py` or pass `--skip-headings-file`.

## 3. Evaluate

```bash
python main.py --lang jv --method cvs
python main.py --lang jv --method all --bert-model LazarusNLP/NusaBERT-base
python main.py --lang id --method bert            # original Indonesian setup
```

Useful options:

| option | meaning |
|---|---|
| `--lang` | wiki code; corpus defaults to `text/` (id) or `data/wiki/<lang>/text` |
| `--bert-model` / `--embed-model` | encoders for `bert` / `bertopic` (default IndoBERT / Qwen3-Embedding) |
| `--strip-heading-echo` | drop the heading sentence wikiextractor repeats at each section start (see below) |
| `--min-headings`, `--min-boundaries`, `--min-sentences` | document filters (defaults = original setup: 3 / 3 / 11) |
| `--max-docs N`, `--max-files N` | quick runs on a seeded sample |
| `--no-save`, `--results-dir`, `--tag` | output control |

Results go to `results/<lang>/<method>/<tag>.pkl`. A merged
`results/<lang>/summary_<tag>.json` collects the scores of all methods.

### Notes

* With default settings, Indonesian parsing gives exactly the same documents as the
  original code. That includes case-sensitive matching of skipped headings. Other
  languages match headings case-insensitively.
* **Heading echo:** wikiextractor writes each heading again as the first line of its
  section (e.g. `Sajarah.`), so every gold segment starts with its title. This is
  the default, to stay comparable with earlier results. Use `--strip-heading-echo`
  for a setting without this leak.
* TextTiling uses NLTK's English stopword list for every language, as in the original setup.
