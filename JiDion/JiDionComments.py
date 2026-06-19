import argparse
import requests
import pandas as pd
from urllib.parse import urlparse, parse_qs


API_KEY = "AIzaSyCOKa3S0j0D7kz1qit****"  # <- put your API key here


def get_video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from a variety of URL formats."""
    parsed_url = urlparse(url)

    if parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query)["v"][0]
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/")[2]
        if parsed_url.path.startswith("/v/"):
            return parsed_url.path.split("/")[2]
    if parsed_url.hostname in ("youtu.be",):
        return parsed_url.path.lstrip("/")

    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_comments(video_id: str, api_key: str, max_pages: int = None):
    """
    Fetch all top-level comments for a video.

    Args:
        video_id: YouTube video ID
        api_key: YouTube Data API v3 key
        max_pages: optional limit for number of pages (None = all)

    Returns:
        List[dict] of comment data
    """
    comments = []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"

    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,        # max allowed by API
        "textFormat": "plainText",
        "key": api_key,
    }

    page_count = 0
    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        for item in data.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "video_id": video_id,
                "author": snippet.get("authorDisplayName"),
                "text": snippet.get("textDisplay"),
                "like_count": snippet.get("likeCount"),
                "published_at": snippet.get("publishedAt"),
                "updated_at": snippet.get("updatedAt"),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })

        page_count += 1
        if max_pages is not None and page_count >= max_pages:
            break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
        params["pageToken"] = next_page_token

    return comments


def collect_comments_to_dataframe(video_url: str, api_key: str, max_pages: int = None) -> pd.DataFrame:
    """High-level helper: URL -> DataFrame of comments."""
    video_id = get_video_id_from_url(video_url)
    comments = fetch_comments(video_id, api_key, max_pages=max_pages)
    df = pd.DataFrame(comments)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect YouTube comments via YouTube Data API v3 and save to CSV.")
    parser.add_argument("-i", "--input", required=True, help="YouTube video URL")
    parser.add_argument("-o", "--output", default="JiDionComments.csv", help="Output CSV filename")
    parser.add_argument("--max-pages", type=int, default=None, help="Maximum API pages to fetch (100 comments/page). Default: all.")
    args = parser.parse_args()

    if API_KEY == "YOUR_YOUTUBE_DATA_API_KEY_HERE":
        raise RuntimeError("Please set your API key in the API_KEY variable.")

    df_comments = collect_comments_to_dataframe(args.input, API_KEY, max_pages=args.max_pages)
    print(f"Fetched {len(df_comments)} comments.")

    # Save to CSV
    df_comments.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Comments saved to {args.output}")
