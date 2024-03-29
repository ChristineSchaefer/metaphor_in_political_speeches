from pathlib import Path

from pydantic import BaseModel, Field
import xml.etree.ElementTree as ET

from src.data_handler.models.politician import Politician
from src.data_handler.models.speech import Speech
from src.utils.database import save_many


class XMLReaderController(BaseModel):
    """
        Controller class for managing and handling XML files.
        This class is responsible for parsing XML files from a specified path and extracting speech and politician data.
    """
    path: str = Field(..., description="path to the folder with xml files")
    politician: Politician | None = Field(default=None, description="politician object")
    speeches: list[dict] = Field(default_factory=list, description="list of speech from politician")

    def parse(self):
        """
            Method that parses XML files in a specified directory.
            It reads each XML file, extracts the person's name, and text,
            and then appends a Speech object to the speeches list.
        """
        path = Path(self.path)
        files = list(path.glob("*.xml"))
        for file in files:
            tree = ET.parse(file)
            root = tree.getroot()
            for child in root:
                self.politician = Politician(name=child.get("person"))
                self._set_party()
                speech = child.find("rohtext").text
                if self.politician:
                    self.speeches.append(Speech(speaker=self.politician, text=speech, url=file.name).dict(exclude={"id"}))
            print(f"With {file.name}: {len(self.speeches)}")
        save_many(Speech.collection(), self.speeches)
        print(f"Add {len(self.speeches)} speeches to database.")

    def _set_party(self):
        """
            Private method to set the party for a Politician object.
            It searches for the politician's name in the database, and if found, sets the politician's party.
            If not found, the politician object is set to None.
        """
        results = Politician.find({"name": {"$in": [self.politician.name]}})
        if len(results) > 0:
            self.politician.party = results[0].party
        else:
            self.politician = None
