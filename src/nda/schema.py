from pydantic import BaseModel, Field


class Party(BaseModel):
    name: str = Field(..., description="Name of one party involved in the contract.")


class NDA(BaseModel):
    """
    * In attribute values, all spaces ` ` and colons `:` should be replaced with an underscore `_`
    * The effective date should be returned in `YYYY-MM-DD` format
    * Values for the attribute `term` should be normalized with the same original units, for example:
        * `eleven months` is changed to `11_months`;
        * all of them should be in the same format: `{number}_{units}`
    """

    effective_date: str = Field(
        ...,
        description="Date in `YYYY-MM-DD` format, at which point the contract is legally binding.",
    )
    jurisdiction: str = Field(
        ...,
        description="Under which state _or_ country jurisdiction is the contract signed.",
    )
    party: list[Party] = Field(
        ..., description="Party or parties involved in the contract."
    )
    term: str = Field(
        ..., description="Length of the legal contract as expressed in the document."
    )
