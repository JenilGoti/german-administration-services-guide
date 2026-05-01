import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


class AgentPageScraperTool:
    """
    Scrapes one page and returns LLM-ready content.

    Service-BW pages are JavaScript apps, so this tool uses the Service-BW
    REST endpoints when it sees a known Service-BW situation or service URL.
    Other URLs are scraped as normal HTML pages.
    """

    SERVICE_BW_BASE_URL = "https://www.service-bw.de"
    SERVICE_BW_API_URL = "https://www.service-bw.de/rest/api"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; GraphAgentScraper/1.0)",
                "Accept": "text/html,application/json",
            }
        )

    # ------------------------
    # PUBLIC TOOL METHOD
    # ------------------------
    def scrape(self, url: str) -> Dict[str, Any]:
        source = self._detect_service_bw_source(url)

        if source and source["source_type"] == "service":
            return self._scrape_service_bw_service(url, source["id"])

        if source and source["source_type"] in {"situation", "sub_situation"}:
            return self._scrape_service_bw_situation(url, source["id"], source["source_type"])

        return self._scrape_html_page(url)

    # ------------------------
    # SERVICE-BW DETECTION
    # ------------------------
    def _detect_service_bw_source(self, url: str) -> Optional[Dict[str, str]]:
        parsed = urlparse(url)
        if "service-bw.de" not in parsed.netloc:
            return None

        service_match = re.search(r"/(?:zufi/)?leistungen/(\d+)", parsed.path)
        if service_match:
            return {"source_type": "service", "id": service_match.group(1)}

        situation_match = re.search(r"/(?:zufi/)?lebenslagen/(\d+)", parsed.path)
        if situation_match:
            return {"source_type": "situation", "id": situation_match.group(1)}

        return None

    # ------------------------
    # SERVICE-BW SERVICE API
    # ------------------------
    def _scrape_service_bw_service(self, url: str, service_id: str) -> Dict[str, Any]:
        api_url = f"{self.SERVICE_BW_API_URL}/leistungen/{service_id}"
        data = self._get_json(api_url)

        sections = []
        all_text_parts = [data.get("name", "")]

        for index, block in enumerate(self._safe_list(data.get("textbloecke"))):
            html = block.get("text", "")
            text = self._html_to_text(html)
            section = {
                "id": self._make_id("service-section", service_id, block.get("typ"), index),
                "type": block.get("typ", ""),
                "title": block.get("titel") or block.get("typ", ""),
                "text": text,
                "html": html,
                "order": index,
            }
            sections.append(section)
            all_text_parts.append(f"{section['title']}\n{section['text']}")

        forms = [
            {
                "id": str(item.get("id") or self._make_id("form", service_id, item.get("name"), item.get("url"))),
                "name": item.get("name", ""),
                "url": item.get("url", ""),
                "type": item.get("typ", ""),
            }
            for item in self._safe_list(data.get("formulare"))
        ]

        authorities = [
            {
                "id": str(item.get("id") or self._make_id("authority", item.get("name"), item.get("url"))),
                "name": item.get("name", ""),
                "type": item.get("typ", ""),
                "address": item.get("adresse", ""),
                "phone": item.get("telefon", ""),
                "email": item.get("email", ""),
                "locationContext": "",
            }
            for item in self._safe_list(data.get("organisationseinheiten"))
        ]

        processes = [
            {
                "id": str(item.get("id") or self._make_id("process", service_id, index, item.get("name"))),
                "title": item.get("name", ""),
                "description": item.get("beschreibung", ""),
                "order": index,
                "channel": item.get("typ", ""),
            }
            for index, item in enumerate(self._safe_list(data.get("prozesse")))
        ]

        raw_text = "\n\n".join(part for part in all_text_parts if part)

        return {
            "status": "success",
            "sourceType": "service",
            "sourceId": service_id,
            "url": url,
            "apiUrl": api_url,
            "title": data.get("name", ""),
            "rawText": raw_text,
            "textHash": self._make_id(raw_text),
            "scrapedAt": datetime.utcnow().isoformat(),
            "service": {
                "id": service_id,
                "name": data.get("name", ""),
                "url": url,
                "description": self._first_section_text(sections),
                "source": "service-bw",
                "regionalisierbar": data.get("regionalisierbar"),
                "mandant": data.get("mandant", ""),
            },
            "serviceSections": sections,
            "forms": forms,
            "authorities": authorities,
            "processSteps": processes,
            "legalBasis": self._extract_legal_basis(service_id, sections),
            "llmInput": self._build_llm_input(
                source_type="service",
                source_id=service_id,
                title=data.get("name", ""),
                url=url,
                text=raw_text,
            ),
        }

    # ------------------------
    # SERVICE-BW SITUATION API
    # ------------------------
    def _scrape_service_bw_situation(
        self,
        url: str,
        situation_id: str,
        source_type: str,
    ) -> Dict[str, Any]:
        api_url = f"{self.SERVICE_BW_API_URL}/lebenslagen/{situation_id}"
        data = self._get_json(api_url)

        tree = data.get("lebenslagenbaum") or {}
        children = self._safe_list(tree.get("untergeordneteLebenslagen"))
        services = self._safe_list(data.get("leistungen"))

        text_parts = [data.get("name", "")]
        for block in self._safe_list(data.get("textbloecke")):
            title = block.get("titel") or block.get("typ") or ""
            text = self._html_to_text(block.get("text", ""))
            if text:
                text_parts.append(f"{title}\n{text}")

        raw_text = "\n\n".join(part for part in text_parts if part)

        return {
            "status": "success",
            "sourceType": source_type,
            "sourceId": situation_id,
            "url": url,
            "apiUrl": api_url,
            "title": data.get("name", ""),
            "rawText": raw_text,
            "textHash": self._make_id(raw_text),
            "scrapedAt": datetime.utcnow().isoformat(),
            "situation": {
                "id": situation_id,
                "name": data.get("name", ""),
                "url": url,
                "description": self._html_blocks_to_text(data.get("textbloecke")),
            },
            "subSituations": [
                {
                    "id": str(child.get("id")),
                    "name": child.get("name", ""),
                    "url": f"{self.SERVICE_BW_BASE_URL}/zufi/lebenslagen/{child.get('id')}",
                    "description": "",
                    "isSelf": False,
                }
                for child in children
                if child.get("id")
            ],
            "services": [
                {
                    "id": str(service.get("id")),
                    "name": service.get("name", ""),
                    "url": f"{self.SERVICE_BW_BASE_URL}/zufi/leistungen/{service.get('id')}",
                    "description": "",
                    "source": "service-bw",
                }
                for service in services
                if service.get("id")
            ],
            "llmInput": self._build_llm_input(
                source_type=source_type,
                source_id=situation_id,
                title=data.get("name", ""),
                url=url,
                text=raw_text,
            ),
        }

    # ------------------------
    # GENERIC HTML SCRAPER
    # ------------------------
    def _scrape_html_page(self, url: str) -> Dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.extract()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        raw_text = soup.get_text("\n", strip=True)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

        links = []
        for a in soup.find_all("a", href=True):
            text = a.get_text(" ", strip=True)
            href = urljoin(url, a["href"])
            if href and text:
                links.append({"text": text, "url": href})

        return {
            "status": "success",
            "sourceType": "web_page",
            "sourceId": self._make_id(url),
            "url": url,
            "apiUrl": None,
            "title": title,
            "rawText": raw_text,
            "textHash": self._make_id(raw_text),
            "scrapedAt": datetime.utcnow().isoformat(),
            "links": links,
            "llmInput": self._build_llm_input(
                source_type="web_page",
                source_id=self._make_id(url),
                title=title,
                url=url,
                text=raw_text,
            ),
        }

    # ------------------------
    # HELPERS
    # ------------------------
    def _get_json(self, url: str) -> Dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout, headers={"Accept": "application/json"})
        response.raise_for_status()
        return response.json()

    def _safe_list(self, value):
        return value if isinstance(value, list) else []

    def _make_id(self, *parts) -> str:
        text = "::".join(str(part) for part in parts if part is not None)
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _html_to_text(self, html: str) -> str:
        if not html:
            return ""
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()

    def _html_blocks_to_text(self, blocks: Any) -> str:
        parts = []
        for block in self._safe_list(blocks):
            text = self._html_to_text(block.get("text", ""))
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _first_section_text(self, sections: List[Dict[str, Any]]) -> str:
        for section in sections:
            if section.get("type") == "preamble" and section.get("text"):
                return section["text"]
        return sections[0]["text"] if sections else ""

    def _extract_legal_basis(self, service_id: str, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        legal_items = []
        for section in sections:
            if section.get("type") == "rechtsgrundlage" and section.get("text"):
                legal_items.append(
                    {
                        "id": self._make_id("legal-basis", service_id, section["text"]),
                        "title": section.get("title", "Rechtsgrundlage"),
                        "lawCode": "",
                        "paragraph": "",
                        "url": "",
                        "text": section["text"],
                    }
                )
        return legal_items

    def _build_llm_input(
        self,
        source_type: str,
        source_id: str,
        title: str,
        url: str,
        text: str,
    ) -> str:
        payload = {
            "sourceType": source_type,
            "sourceId": source_id,
            "title": title,
            "url": url,
            "text": text,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    tool = AgentPageScraperTool()
    result = tool.scrape("https://www.service-bw.de/zufi/leistungen/172")
    print(result["llmInput"])
