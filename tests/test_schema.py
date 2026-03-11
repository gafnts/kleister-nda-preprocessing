"""
Tests for schema.py
(Pydantic models: Party, NDA)
"""

from typing import cast

import pytest
from pydantic import ValidationError

from nda.schema import NDA, Party


@pytest.fixture
def empty_nda() -> NDA:
    """Return an empty, valid NDA instance."""
    return NDA()


@pytest.fixture
def full_nda() -> NDA:
    """Return a fully-populated, valid NDA instance."""
    return NDA(
        effective_date="2020-01-01",
        jurisdiction="California",
        party=[Party(name="MPB_Corp"), Party(name="Gadget_Ltd")],
        term="2_years",
    )


# ---------------------------------------------------------------------------
# Party
# ---------------------------------------------------------------------------


class TestParty:
    def test_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            Party()  # type: ignore[call-arg]

    def test_valid_party(self) -> None:
        p = Party(name="MPB_Corp")
        assert p.name == "MPB_Corp"


class TestPartyNormalization:
    def test_spaces_become_underscores(self) -> None:
        p = Party(name="MPB Corp")
        assert p.name == "MPB_Corp"

    def test_colons_become_underscores(self) -> None:
        p = Party(name="Corp:LLC")
        assert p.name == "Corp_LLC"

    def test_already_clean_value_unchanged(self) -> None:
        p = Party(name="Acme")
        assert p.name == "Acme"


# ---------------------------------------------------------------------------
# NDA — defaults
# ---------------------------------------------------------------------------


class TestNDADefaults:
    def test_optional_fields_default_to_none(self, empty_nda: NDA) -> None:
        assert empty_nda.effective_date is None
        assert empty_nda.jurisdiction is None
        assert empty_nda.term is None

    def test_party_defaults_to_empty_list(self, empty_nda: NDA) -> None:
        assert empty_nda.party == []

    def test_party_default_is_not_shared_across_instances(self) -> None:
        """Each NDA gets its own list — no mutable default aliasing."""
        a = NDA()
        b = NDA()
        a.party.append(Party(name="X"))
        assert b.party == []


# ---------------------------------------------------------------------------
# NDA — effective_date validation
# ---------------------------------------------------------------------------


class TestEffectiveDateValidation:
    def test_valid_date(self) -> None:
        nda = NDA(effective_date="2020-01-15")
        assert nda.effective_date == "2020-01-15"

    def test_none_is_allowed(self) -> None:
        nda = NDA(effective_date=None)
        assert nda.effective_date is None

    def test_rejects_non_date_string(self) -> None:
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            NDA(effective_date="not-a-date")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            NDA(effective_date="")

    def test_rejects_wrong_separator(self) -> None:
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            NDA(effective_date="2020/01/15")


# ---------------------------------------------------------------------------
# NDA — underscore normalization (jurisdiction, term)
# ---------------------------------------------------------------------------


class TestNDAUnderscoreNormalization:
    def test_jurisdiction_spaces(self) -> None:
        nda = NDA(jurisdiction="New York")
        assert nda.jurisdiction == "New_York"

    def test_jurisdiction_colons(self) -> None:
        nda = NDA(jurisdiction="Region:East")
        assert nda.jurisdiction == "Region_East"

    def test_jurisdiction_already_clean(self) -> None:
        nda = NDA(jurisdiction="California")
        assert nda.jurisdiction == "California"

    def test_jurisdiction_none_is_allowed(self) -> None:
        nda = NDA(jurisdiction=None)
        assert nda.jurisdiction is None

    def test_term_spaces_normalized_before_format_check(self) -> None:
        """normalize_underscores runs before validate_term_format."""
        nda = NDA(term="2 years")
        assert nda.term == "2_years"

    def test_term_colons_normalized(self) -> None:
        nda = NDA(term="2:years")
        assert nda.term == "2_years"


# ---------------------------------------------------------------------------
# NDA — term format validation
# ---------------------------------------------------------------------------


class TestTermValidation:
    def test_valid_term(self) -> None:
        nda = NDA(term="11_months")
        assert nda.term == "11_months"

    def test_none_is_allowed(self) -> None:
        nda = NDA(term=None)
        assert nda.term is None

    def test_rejects_plain_text(self) -> None:
        with pytest.raises(ValidationError, match="number.*units"):
            NDA(term="forever")

    def test_rejects_units_without_number(self) -> None:
        with pytest.raises(ValidationError, match="number.*units"):
            NDA(term="years")


# ---------------------------------------------------------------------------
# NDA — domain edge cases
# ---------------------------------------------------------------------------


class TestNDADomainEdgeCases:
    def test_party_accepts_raw_dicts(self) -> None:
        """Pydantic coerces dicts into Party objects automatically."""
        nda = NDA(party=cast(list[Party], [{"name": "Acme"}]))
        assert isinstance(nda.party[0], Party)
        assert nda.party[0].name == "Acme"

    def test_party_rejects_dict_without_name(self) -> None:
        with pytest.raises(ValidationError):
            NDA(party=cast(list[Party], [{}]))


# ---------------------------------------------------------------------------
# NDA — serialization snapshot
# ---------------------------------------------------------------------------


class TestNDASerialization:
    def test_full_nda_round_trips_through_dump(self, full_nda: NDA) -> None:
        data = full_nda.model_dump()
        reconstructed = NDA(**data)
        assert reconstructed == full_nda

    def test_empty_nda_dump(self, empty_nda: NDA) -> None:
        assert empty_nda.model_dump() == {
            "effective_date": None,
            "jurisdiction": None,
            "party": [],
            "term": None,
        }
