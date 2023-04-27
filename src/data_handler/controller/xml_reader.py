from pydantic import BaseModel
import xml.etree.ElementTree as ET

from src.data_handler.models.speech import Speech


class XMLReaderController(BaseModel):
    path: str
    speeches: list[Speech] = []

    def parse(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        print(root.tag)
        for child in root:
            print(child.tag, child.attrib)
            print(child[0].text)
