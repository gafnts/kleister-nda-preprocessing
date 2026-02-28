from collections import defaultdict
from typing import Any

import pandas as pd

from nda.data_loader import Partition
from nda.schema import NDA, Party


class LabelTransformer:
    @staticmethod
    def transform(df: pd.DataFrame, partition: Partition) -> pd.DataFrame:
        if partition == "test-A":
            return df
        return (
            df.assign(
                labels_canonical=lambda df: df.labels.apply(
                    LabelTransformer.sort_label_fields
                )
            )
            .assign(
                labels_schema=lambda df: df.labels_canonical.apply(
                    LabelTransformer.parse_label_to_schema
                )
            )
            .assign(
                labels_serialized=lambda df: df.labels_schema.apply(
                    LabelTransformer.label_schema_to_string
                )
            )
        )

    @staticmethod
    def sort_label_fields(string: str) -> str:
        schema_order = ["effective_date", "jurisdiction", "party", "term"]

        pairs: list[tuple[str, str]] = []
        for part in string.strip().split():
            if "=" in part:
                key, value = part.split("=", 1)
                pairs.append((key, value))

        grouped: dict[str, list[str]] = {k: [] for k in schema_order}
        others: list[str] = []
        for key, value in pairs:
            if key in grouped:
                grouped[key].append(f"{key}={value}")
            else:
                others.append(f"{key}={value}")

        result: list[str] = []
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
    def label_schema_to_string(nda_dict: dict[str, Any]) -> str:
        parts: list[str] = []

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
