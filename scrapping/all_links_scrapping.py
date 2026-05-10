import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests

SCRAPPING_DIR = Path(__file__).resolve().parent
SERVICE_BW_OUTPUT_PATH = SCRAPPING_DIR / "service_bw_output.json"


@dataclass
class Service:
    id: str
    url: str
    name: Optional[str] = None

    def to_json(self) -> Dict:
        data = {"id": self.id, "url": self.url}
        if self.name:
            data["name"] = self.name
        return data


@dataclass
class SubSituation:
    id: str
    url: str
    name: Optional[str] = None
    is_self: bool = False
    services: List[Service] = field(default_factory=list)

    def add_service(self, service: Service) -> None:
        if service.id not in {item.id for item in self.services}:
            self.services.append(service)

    def to_json(self) -> Dict:
        data = {
            "id": self.id,
            "url": self.url,
            "is_self": self.is_self,
            "services": [service.to_json() for service in self.services],
        }
        if self.name:
            data["name"] = self.name
        return data


@dataclass
class Situation:
    id: str
    url: str
    name: Optional[str] = None
    sub_situations: List[SubSituation] = field(default_factory=list)

    def add_sub_situation(self, sub_situation: SubSituation) -> None:
        if sub_situation.id not in {item.id for item in self.sub_situations}:
            self.sub_situations.append(sub_situation)

    def to_json(self) -> Dict:
        data = {
            "id": self.id,
            "url": self.url,
            "sub_situations": [item.to_json() for item in self.sub_situations],
        }
        if self.name:
            data["name"] = self.name
        return data


class ServiceBWClient:
    BASE_URL = "https://www.service-bw.de"
    API_URL = f"{BASE_URL}/rest/api"

    def __init__(self, delay_seconds: float = 0.1):
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; ServiceBWDataCrawler/1.0)",
            }
        )

    def situation_url(self, situation_id: str) -> str:
        return f"{self.BASE_URL}/zufi/lebenslagen/{situation_id}"

    def service_url(self, service_id: str) -> str:
        return f"{self.BASE_URL}/zufi/leistungen/{service_id}"

    def get_json(self, path: str) -> Dict:
        url = f"{self.API_URL}{path}"
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        time.sleep(self.delay_seconds)
        return response.json()

    def get_main_groups(self) -> List[Dict]:
        return self.get_json("/lebenslagen/gruppen")

    def get_situation(self, situation_id: str) -> Dict:
        return self.get_json(f"/lebenslagen/{situation_id}")


class ServiceBWCrawler:
    def __init__(self, delay_seconds: float = 0.1):
        self.client = ServiceBWClient(delay_seconds=delay_seconds)
        self.situation_cache: Dict[str, Dict] = {}

    def read_situation(self, situation_id: str) -> Dict:
        if situation_id not in self.situation_cache:
            self.situation_cache[situation_id] = self.client.get_situation(situation_id)
        return self.situation_cache[situation_id]

    def iter_tree(self, node: Dict) -> Iterable[Dict]:
        yield node
        for child in node.get("untergeordneteLebenslagen") or []:
            yield from self.iter_tree(child)

    def extract_main_situations(self) -> List[Situation]:
        situations: List[Situation] = []
        seen: Set[str] = set()

        for group in self.client.get_main_groups():
            for item in group.get("lebenslagen") or []:
                situation_id = str(item["id"])
                if situation_id in seen:
                    continue
                seen.add(situation_id)
                situations.append(
                    Situation(
                        id=situation_id,
                        name=item.get("name"),
                        url=self.client.situation_url(situation_id),
                    )
                )

        return situations

    def add_services(self, sub_situation: SubSituation) -> None:
        data = self.read_situation(sub_situation.id)

        for item in data.get("leistungen") or []:
            service_id = str(item["id"])
            sub_situation.add_service(
                Service(
                    id=service_id,
                    name=item.get("name"),
                    url=self.client.service_url(service_id),
                )
            )

    def add_sub_situations(self, situation: Situation) -> None:
        data = self.read_situation(situation.id)
        tree = data.get("lebenslagenbaum") or {
            "id": data["id"],
            "name": data.get("name"),
            "untergeordneteLebenslagen": [],
        }

        for node in self.iter_tree(tree):
            sub_id = str(node["id"])
            situation.add_sub_situation(
                SubSituation(
                    id=sub_id,
                    name=node.get("name"),
                    url=self.client.situation_url(sub_id),
                    is_self=(sub_id == situation.id),
                )
            )

    def crawl(self, start_url: str = "https://www.service-bw.de/zufi/lebenslagen") -> Dict:
        situations = self.extract_main_situations()

        for index, situation in enumerate(situations, start=1):
            print(f"[{index}/{len(situations)}] {situation.id} {situation.name or ''}")
            self.add_sub_situations(situation)

            for sub_situation in situation.sub_situations:
                self.add_services(sub_situation)

        return {
            "source": start_url,
            "situations": [situation.to_json() for situation in situations],
            "summary": {
                "situations": len(situations),
                "sub_situations": sum(len(item.sub_situations) for item in situations),
                "services": sum(
                    len(sub.services)
                    for situation in situations
                    for sub in situation.sub_situations
                ),
            },
        }

    def save_json(self, data: Dict, filename: str | Path = SERVICE_BW_OUTPUT_PATH) -> str:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        return str(output_path)


def main():
    crawler = ServiceBWCrawler()
    output = crawler.crawl()
    file_path = crawler.save_json(output)

    print("Saved:", file_path)
    print(json.dumps(output["summary"], indent=2, ensure_ascii=False))
    return file_path
