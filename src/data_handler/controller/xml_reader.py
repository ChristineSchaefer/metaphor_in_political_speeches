from pathlib import Path

from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET

from src.data_handler.models.politician import Politician
from src.data_handler.models.speech import Speech
from src.utils.database import save_many
from src.utils.text_handler import normalize


class XMLReaderController(BaseModel):
    path: str
    politician: Politician | None = Field(default_factory=None)
    speeches: list[dict] = Field(default_factory=list)

    def parse(self):
        path = Path(self.path)
        files = list(path.glob("*.xml"))
        for file in files:
            tree = ET.parse(file)
            root = tree.getroot()
            for child in root:
                self.politician = Politician(name=child.get("person"))
                self._set_party()
                speech = normalize(child.find("rohtext").text)
                if self.politician:
                    self.speeches.append(Speech(speaker=self.politician, text=speech, url=file.name).dict(exclude={"id"}))
            print(f"With {file.name}: {len(self.speeches)}")
        save_many(Speech.collection(), self.speeches)
        print(f"Add {len(self.speeches)} speeches to database.")

    def _set_party(self):
        results = Politician.find({"name": {"$in": [self.politician.name]}})
        if len(results) > 0:
            self.politician.party = results[0].party
        else:
            self.politician = None
