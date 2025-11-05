import time
from datetime import datetime
from urllib.parse import urlparse
from langchain_community.utilities import GoogleSerperAPIWrapper
from firecrawl import Firecrawl
import os
import json

# Set your Serper API key
os.environ["SERPER_API_KEY"] = "3cf13dc86ab5a8f0c9bd7dfc8fdb351766aa973a"

# Initialize Serper API and Firecrawl clients with your API keys
serper = GoogleSerperAPIWrapper()
firecrawl = Firecrawl(api_key="fc-5d75931ac4f546d3b8a9f9f6c1dd61a8")


def is_valid_url(u):
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except:
        return False


def search_serper(keyword, top_k=2):
    urls = []
    print(f"Searching Serper for keyword: {keyword}")
    try:
        result_json = serper.results(keyword)
        print(f"Serper raw result keys: {result_json.keys()}")

        if 'organic' in result_json:
            for organic_result in result_json['organic']:
                if 'link' in organic_result and is_valid_url(organic_result['link']):
                    urls.append(organic_result['link'])
                    if len(urls) >= top_k:
                        break
    except Exception as e:
        print(f"Error during Serper search: {e}")
    print(f"Found URLs: {urls}")
    return urls


def fetch_with_firecrawl(url):
    print(f"Fetching with Firecrawl: {url}")
    try:
        response = firecrawl.scrape(url, formats=["markdown"])
        markdown_text = response.markdown

        # 🚫 Remove image markdown lines
        markdown_text = "\n".join(
            line for line in markdown_text.splitlines()
            if not line.strip().startswith("![](") and not line.strip().startswith("![")
        )

        print(f"Firecrawl fetched text length (after removing images): {len(markdown_text)}")
        lines = markdown_text.split('\n')
        title = ""
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break
        print(f"Extracted title: {title}")
        return title, markdown_text
    except Exception as e:
        print(f"Error during Firecrawl fetch: {e}")
        return "", ""


def scout(subqs, top_k=1, output_file="scout_results.txt"):
    print(f"Starting scout with subqueries: {subqs}")
    results = []
    for sq in subqs:
        sid, stext = sq["id"], sq["text"]
        keywords = sq.get("search_keywords") or [stext]
        print(f"Processing subquery ID: {sid}, Text: {stext}, Keywords: {keywords}")
        for kw in keywords[:2]:
            urls = search_serper(kw, top_k=top_k)
            for url in urls:
                title, text = fetch_with_firecrawl(url)
                if title or text:
                    result = {
                        "subq_id": sid,
                        "subq_text": stext,
                        "keyword": kw,
                        "url": url,
                        "title": title,
                        "text": text,
                        "fetched_at": datetime.utcnow().isoformat() + "Z"
                    }
                    results.append(result)
                time.sleep(0.5)

    # 🔥 Save results to a .txt file
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Subquery ID: {r['subq_id']}\n")
            f.write(f"Subquery Text: {r['subq_text']}\n")
            f.write(f"Keyword: {r['keyword']}\n")
            f.write(f"URL: {r['url']}\n")
            f.write(f"Title: {r['title']}\n")
            f.write(f"Fetched At: {r['fetched_at']}\n")
            f.write("Content:\n")
            f.write(r["text"][:2000])  # limit to first 2000 chars for readability
            f.write("\n" + "="*80 + "\n\n")

    print(f"Scout completed. Total results found: {len(results)}")
    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    demo_subs = [{"id": "Q1", "text": "H1B visa rules 2025", "search_keywords": ["H1B rules 2025"]}]
    res = scout(demo_subs)
    for r in res:
        print(r)
