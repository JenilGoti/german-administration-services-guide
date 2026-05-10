import requests
from bs4 import BeautifulSoup


class WebScraper:

    def scrape(self, url: str):

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Research Assistant Bot)"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")

            # remove junk
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            text = soup.get_text(separator=" ", strip=True)

            clean_text = " ".join(text.split())

            return {
                "url": url,
                "content": clean_text[:8000],
                "status": "success"
            }

        except Exception as e:
            return {
                "url": url,
                "status": "error",
                "error": str(e)
            }