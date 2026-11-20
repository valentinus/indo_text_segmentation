"""Collect public top-level comments from a YouTube video (YouTube Data API v3).

Usage:
  export YOUTUBE_API_KEY=...        # Google Cloud console > APIs > YouTube Data API v3
  python scripts/collect_youtube_comments.py VIDEO_ID --max 300 --out data/real_comments.csv

Writes id, comment, like_count, published_at. Author names are NOT saved
(keep classroom data anonymous). Check YouTube's Terms of Service and your
institution's research-ethics rules before using or sharing collected comments.
"""
import argparse
import csv
import json
import os
import urllib.parse
import urllib.request

API = "https://www.googleapis.com/youtube/v3/commentThreads"


def fetch(video_id: str, key: str, max_items: int):
    out, token = [], None
    while len(out) < max_items:
        params = {"part": "snippet", "videoId": video_id, "maxResults": 100,
                  "textFormat": "plainText", "order": "relevance", "key": key}
        if token:
            params["pageToken"] = token
        with urllib.request.urlopen(API + "?" + urllib.parse.urlencode(params)) as r:
            data = json.load(r)
        for item in data.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            out.append({"comment": s["textDisplay"].replace("\n", " ").strip(),
                        "like_count": s.get("likeCount", 0),
                        "published_at": s.get("publishedAt", "")})
        token = data.get("nextPageToken")
        if not token:
            break
    return out[:max_items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--max", type=int, default=300)
    ap.add_argument("--out", default="data/real_comments.csv")
    a = ap.parse_args()
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise SystemExit("Set YOUTUBE_API_KEY first.")
    rows = fetch(a.video_id, key, a.max)
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "comment", "like_count", "published_at"])
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            w.writerow({"id": i, **r})
    print(f"saved {len(rows)} comments to {a.out}")


if __name__ == "__main__":
    main()
