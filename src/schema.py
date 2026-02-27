from pydantic import BaseModel, Field


class NDA(BaseModel):
    id: str = Field(..., description="Unique identifier for the NDA")
    title: str = Field(..., description="Title of the NDA")
    parties: list[str] = Field(..., description="List of parties involved in the NDA")
    effective_date: str = Field(..., description="Effective date of the NDA")
    expiration_date: str = Field(..., description="Expiration date of the NDA")
    governing_law: str = Field(..., description="Governing law of the NDA")
