from pydantic import BaseModel


class Speech(BaseModel):
    text: str
    speaker: str
    party: str

    class Config:
        collection_name = "speeches"
