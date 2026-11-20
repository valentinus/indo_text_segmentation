"""Illustrative YouTube-comment dataset for the RDSP 2026 guest lecture.

Case: comments under a travel video about Fushimi Inari Taisha (Kyoto).
The comments are COMPOSED FOR TEACHING: they imitate the style, topics and
languages of real comments (EN / JA / ID) but are not copied from YouTube and
contain no usernames. Swap in real data with ../scripts/collect_youtube_comments.py.

Each row: id, comment, lang, English gloss (for JA/ID), and the labels produced
by the LLM annotator (Claude, codebook prompt in ../prompts.md):
primary topic, optional secondary topic, sentiment.
"""

TOPICS = {
    "SCN": "Scenery & atmosphere",
    "CRW": "Crowds & overtourism",
    "CUL": "Culture, history & etiquette",
    "PRC": "Practical tips & access",
    "FDS": "Food & shopping",
    "VID": "About the video / creator",
    "OTH": "Other (spam, personal, off-topic)",
}
SENTIMENTS = ["Positive", "Neutral", "Negative"]

# id, comment, lang, gloss, topic, secondary, sentiment
ROWS = [
    (1, "Went at 6am and had the whole mountain almost to myself. The orange gates glowing in the morning light were unreal.", "en", "", "SCN", "PRC", "Positive"),
    (2, "The first 15 minutes of gates are packed shoulder to shoulder, you can't even take a photo without 50 strangers in it.", "en", "", "CRW", "", "Negative"),
    (3, "Tip: keep walking past the Yotsutsuji intersection. 90% of tourists turn back there and the upper trail is peaceful.", "en", "", "PRC", "CRW", "Positive"),
    (4, "The kitsune statues holding keys in their mouths always give me chills. Such a sacred atmosphere.", "en", "", "CUL", "SCN", "Positive"),
    (5, "Inari station on the JR Nara line is literally right in front of the gate, super easy from Kyoto station.", "en", "", "PRC", "", "Positive"),
    (6, "Kitsune udon and inari sushi at the little shop near the entrance were so good after the hike!", "en", "", "FDS", "", "Positive"),
    (7, "Your drone shots are amazing, what camera do you use?", "en", "", "VID", "", "Positive"),
    (8, "Not worth it anymore. Too many tour groups, selfie sticks everywhere and people blocking the path.", "en", "", "CRW", "", "Negative"),
    (9, "Each torii is donated by a company or person, their name and date are written on the back. Some are over 100 years old.", "en", "", "CUL", "", "Neutral"),
    (10, "Is it open at night? I'm arriving in Kyoto around 9pm.", "en", "", "PRC", "", "Neutral"),
    (11, "伏見稲荷は夜に行くと本当に静かで、ちょっと怖いけど幻想的でした。", "ja", "At night Fushimi Inari is really quiet - a little scary, but magical.", "SCN", "", "Positive"),
    (12, "Bagus banget pemandangannya, tapi capek naik tangganya sampai puncak 😅", "id", "The view is really great, but climbing the stairs to the top is exhausting.", "SCN", "PRC", "Positive"),
    (13, "Great, another 'hidden gem' video that will bring 10,000 more people tomorrow 🙄", "en", "", "CRW", "VID", "Negative"),
    (14, "The hike to the summit takes about 2-3 hours round trip, bring water and good shoes.", "en", "", "PRC", "", "Neutral"),
    (15, "Locals in my neighborhood complain the narrow streets are blocked by buses every morning. Overtourism is real.", "en", "", "CRW", "", "Negative"),
    (16, "Make sure to bow before passing through the main gate and don't walk in the center, that path is for the gods.", "en", "", "CUL", "PRC", "Neutral"),
    (17, "Check out my channel for more Japan travel vlogs!!! 🔥🔥", "en", "", "OTH", "", "Neutral"),
    (18, "The street food at the entrance is overpriced but the grilled mochi was worth it.", "en", "", "FDS", "", "Neutral"),
    (19, "Saya ke sini waktu musim gugur 2023, daun merahnya cantik sekali di antara gerbang oranye.", "id", "I came here in autumn 2023; the red leaves between the orange gates were so beautiful.", "SCN", "", "Positive"),
    (20, "Music in this video is too loud, couldn't hear the narration.", "en", "", "VID", "", "Negative"),
    (21, "Honestly the view from the top is a bit disappointing, mostly trees. The gates are the real highlight.", "en", "", "SCN", "", "Negative"),
    (22, "Foxes are messengers of Inari, the god of rice and prosperity. That's why merchants pray here for business success.", "en", "", "CUL", "", "Neutral"),
    (23, "外国人観光客が多すぎて、地元の参拝客がゆっくりお参りできないのが残念。", "ja", "Too many foreign tourists - it's a shame local worshippers can't pray in peace.", "CRW", "CUL", "Negative"),
    (24, "Came here on my honeymoon 10 years ago. Watching this brought back so many memories ❤️", "en", "", "OTH", "VID", "Positive"),
    (25, "Wear comfortable shoes, the stone steps get slippery when it rains.", "en", "", "PRC", "", "Neutral"),
    (26, "Bought a small fox ema to write my wish on. Such a nice souvenir and only 500 yen.", "en", "", "FDS", "CUL", "Positive"),
    (27, "Lebih baik datang pagi-pagi, siang hari ramai sekali dan susah jalan.", "id", "Better to come early in the morning; at midday it's very crowded and hard to walk.", "PRC", "CRW", "Neutral"),
    (28, "This is the best travel video about Kyoto I've seen, great editing!", "en", "", "VID", "", "Positive"),
    (29, "The rows of gates feel like walking through a tunnel of fire. Absolutely breathtaking.", "en", "", "SCN", "", "Positive"),
    (30, "Tourists ignoring the 'no photos' sign at the inner shrine was really disrespectful.", "en", "", "CRW", "CUL", "Negative"),
    (31, "Entry is free, which is amazing for such a famous place.", "en", "", "PRC", "", "Positive"),
    (32, "参道の屋台のいなり寿司、甘くて美味しかった！", "ja", "The inari sushi from the stalls on the approach was sweet and delicious!", "FDS", "", "Positive"),
    (33, "Has anyone compared this with Itsukushima? Which one should I visit with only one day?", "en", "", "PRC", "", "Neutral"),
    (34, "The quiet bamboo section near the top with cats sleeping on the stones was my favourite part.", "en", "", "SCN", "", "Positive"),
    (35, "Why do people say it's spiritual? It's just a bunch of painted gates for Instagram now.", "en", "", "CUL", "CRW", "Negative"),
    (36, "Kyoto should introduce a tourist tax or reservation system like Machu Picchu.", "en", "", "CRW", "", "Neutral"),
    (37, "Sound of crows and wind in the trees at dusk, so peaceful.", "en", "", "SCN", "", "Positive"),
    (38, "Mantap videonya, kualitas 4K jernih banget 👍", "id", "Great video, the 4K quality is super clear.", "VID", "", "Positive"),
    (39, "The shrine dates back to 711 AD, long before Kyoto became the capital.", "en", "", "CUL", "", "Neutral"),
    (40, "Tried the fox-shaped senbei crackers, cute but tasted like nothing lol", "en", "", "FDS", "", "Negative"),
    (41, "Went during Golden Week. Big mistake. Two hours in a human traffic jam.", "en", "", "CRW", "PRC", "Negative"),
    (42, "There's a vending machine at almost every rest stop on the trail, so no need to carry too much.", "en", "", "PRC", "", "Neutral"),
    (43, "Praying at the main hall and walking the Senbon Torii made me feel calm and grateful.", "en", "", "CUL", "SCN", "Positive"),
    (44, "I love how you included the history at the beginning, very informative.", "en", "", "VID", "CUL", "Positive"),
    (45, "Sungguh tempat yang sakral, harus tetap menjaga sopan santun saat berkunjung.", "id", "Truly a sacred place; visitors must keep good manners.", "CUL", "", "Neutral"),
    (46, "Free crypto giveaway, click the link in my bio", "en", "", "OTH", "", "Neutral"),
    (47, "Lighting in the evening part of this video was weird, too dark to see the gates.", "en", "", "VID", "", "Negative"),
    (48, "Five years ago it wasn't this crowded. Now it feels like Disneyland.", "en", "", "CRW", "", "Negative"),
    (49, "City bus Minami 5 also stops here, but trains are much faster during rush hour.", "en", "", "PRC", "", "Neutral"),
    (50, "The moss on the old stone lanterns gives the whole place a magical, ancient feeling.", "en", "", "SCN", "CUL", "Positive"),
    (51, "Matcha soft serve near the station 10/10 would climb the mountain again for it", "en", "", "FDS", "", "Positive"),
    (52, "早朝に行くのがおすすめ。8時を過ぎると一気に混みます。", "ja", "I recommend going early in the morning. After 8 it suddenly gets crowded.", "PRC", "CRW", "Neutral"),
    (53, "The small sub-shrines with mini torii stacked everywhere are fascinating, each one is someone's prayer.", "en", "", "CUL", "", "Positive"),
    (54, "Respect the locals. Don't eat while walking on the shrine path, it's considered rude.", "en", "", "CUL", "PRC", "Neutral"),
    (55, "Why didn't you show the summit? Felt like the video ended too early.", "en", "", "VID", "", "Negative"),
    (56, "Ramai banget, kayak pasar. Nggak bisa menikmati suasananya.", "id", "So crowded, like a market. You can't enjoy the atmosphere.", "CRW", "", "Negative"),
    (57, "Sunset from the Yotsutsuji viewpoint over Kyoto city is stunning.", "en", "", "SCN", "", "Positive"),
    (58, "Gift shop prices are crazy, 1500 yen for a tiny keychain.", "en", "", "FDS", "", "Negative"),
    (59, "Literally the most beautiful place I have ever seen in my life.", "en", "", "SCN", "", "Positive"),
    (60, "Hey, do you know if there are lockers for luggage at Inari station?", "en", "", "PRC", "", "Neutral"),
]

# The 20 comments used in the in-class human annotation activity
# (mix of easy and deliberately ambiguous items).
ANNOTATION_SET = [1, 2, 3, 4, 6, 8, 12, 13, 16, 18, 23, 24, 26, 27, 30, 35, 36, 40, 44, 54]
