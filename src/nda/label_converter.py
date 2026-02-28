import pandas as pd

from collections import defaultdict
from typing import List, Tuple, Any, Dict

from nda.schema import NDA, Party
from nda.data_loader import Partition


class LabelConverter:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def convert_labels(self, partition: Partition = "train") -> pd.DataFrame:
        if partition == "test-A":
            return self.df
        return (
            self.df.assign(
                labels_canonical=lambda df: df.labels.apply(self.sort_label_fields)
            )
            .assign(
                labels_schema=lambda df: df.labels_canonical.apply(
                    self.parse_label_to_schema
                )
            )
            .assign(
                labels_serialized=lambda df: df.labels_schema.apply(
                    self.label_schema_to_string
                )
            )
        )

    @staticmethod
    def sort_label_fields(string: str) -> str:
        schema_order = ["effective_date", "jurisdiction", "party", "term"]

        pairs: List[Tuple[str, str]] = []
        for part in string.strip().split():
            if "=" in part:
                key, value = part.split("=", 1)
                pairs.append((key, value))

        grouped: dict[str, List[str]] = {k: [] for k in schema_order}
        others: List[str] = []
        for key, value in pairs:
            if key in grouped:
                grouped[key].append(f"{key}={value}")
            else:
                others.append(f"{key}={value}")

        result: List[str] = []
        for key in schema_order:
            result.extend(grouped[key])
        result.extend(others)
        return " ".join(result)

    @staticmethod
    def parse_label_to_schema(string: str) -> dict[str, Any]:
        result: defaultdict[str, list[str]] = defaultdict(list)

        for token in string.strip().split():
            key, _, value = token.partition("=")
            result[key].append(value)

        effective_date = result.get("effective_date", [""])[0]
        jurisdiction = result.get("jurisdiction", [""])[0]
        term = result.get("term", [None])[0]
        parties = result.get("party", [])
        party = [Party(name=p) for p in parties]

        nda = NDA(
            effective_date=effective_date,
            jurisdiction=jurisdiction,
            term=term,
            party=party,
        )

        return nda.model_dump()

    @staticmethod
    def label_schema_to_string(nda_dict: Dict[str, Any]) -> str:
        parts: List[str] = []

        for key in ["effective_date", "jurisdiction"]:
            value = nda_dict.get(key)
            if value:
                parts.append(f"{key}={value}")

        for party in nda_dict.get("party", []):
            name = party.get("name")
            if name:
                parts.append(f"party={name}")

        term = nda_dict.get("term")
        if term:
            parts.append(f"term={term}")

        return " ".join(parts)
