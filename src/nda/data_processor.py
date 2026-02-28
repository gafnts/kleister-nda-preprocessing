from typing import Any
from collections import defaultdict
from nda.schema import NDA, Party


def parse_to_json(input_string: str) -> dict[str, Any]:
    result: defaultdict[str, list[str]] = defaultdict(list)

    for token in input_string.strip().split():
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
