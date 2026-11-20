# Prompt sheet — Activity 2 (AI annotation)

Use with ChatGPT, Gemini or Claude. Paste the comments from
`data/fushimi_inari_comments.csv` (or the 20 comments from the worksheet) under the prompt.
Do not paste usernames or personal data.

---

## Prompt A — zero-shot (no codebook)

```
Here are YouTube comments on a travel video about Fushimi Inari Taisha, Kyoto.
What topic is each comment about? Also give its sentiment.
```

*Watch for:* labels change from run to run, the model invents its own categories,
and the output can't be compared with the human labels.

## Prompt B — codebook prompt (used to produce the AI labels in the lecture)

```
You are an annotator for a tourism research project.
The texts are YouTube comments on a travel video about Fushimi Inari Taisha (Kyoto).
Comments may be in English, Japanese or Indonesian.

Assign each comment exactly ONE primary topic from this codebook:
SCN = Scenery & atmosphere (views, gates, light, nature, mood)
CRW = Crowds & overtourism (visitor numbers, tourist behaviour, impact on locals, policy)
CUL = Culture, history & etiquette (religion, symbols, rituals, manners)
PRC = Practical tips & access (transport, timing, hours, cost, route, what to bring)
FDS = Food & shopping (food, drinks, souvenirs, shop prices)
VID = About the video / creator (editing, music, camera, content)
OTH = Other (spam, self-promotion, personal stories with no place content)

If a second topic is clearly present, give it as "secondary", else "-".
Sentiment: POS, NEU (facts, questions, tips, mixed) or NEG. Sarcasm counts by its intended meaning.

Return a table: id | primary | secondary | sentiment | confidence (high/medium/low) | reason (max 12 words).
Do not skip or merge comments. Do not invent comments.
```

## Prompt C — explain the disagreements

```
Three human annotators and an AI labelled the comments below with the same codebook.
For each comment where they disagree, explain in one sentence what makes it ambiguous
(e.g., sarcasm, two topics, cultural knowledge needed, missing context).
Then suggest one rule we could add to the codebook to reduce this disagreement.
```

## Prompt D — from labels to insight

```
I work for the Kyoto city tourism office. Goal: improve the visitor experience at
Fushimi Inari while protecting residents and worshippers.
Below are topic and sentiment counts from 60 comments (and example comments).
Write: (1) a 3-sentence executive summary, (2) 3 insights, each with its supporting
number, (3) 3 actions in priority order. Quote only comments that appear in the data.
```

## Prompt E — self-check

```
Re-check your table: count the rows (should be N), list any comment ids you skipped,
and flag every label with low confidence.
```

---

### Quality checks before trusting AI labels
1. Row count = input count, ids unchanged.
2. Compare with the human majority on a sample (the worksheet does this in `4_AI_labels`).
3. Run the same prompt twice — labels that flip are "unstable" and need a human.
4. Every quoted comment must exist in the data (hallucination check).
