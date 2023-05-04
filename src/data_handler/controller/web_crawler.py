import os

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.config import BASE_DIR
from src.data_handler.models.politician import Politician
from src.data_handler.models.speech import Speech


def normalize(text: str):
    for char in '()"=+-.,\n':
        text = text.replace(char, "")
    return text


def remove_string(token: list, to_remove: list):
    for i, t in enumerate(token):
        if t in to_remove:
            token[i] = ""
    return [x for x in token if x]


def write_to_txt(text: str, name: str, mode: str):
    file = open(os.path.join(BASE_DIR, name), mode)
    file.write(text + "\n")
    file.close()


def read_from_txt(path: str) -> list:
    content = []
    with open(os.path.join(BASE_DIR, path)) as f:
        lines = f.readlines()
        for line in lines:
            content.append(normalize(line).split())
    return content


class CrawlerController(BaseModel):
    url: str
    index: int
    speeches: list[Speech] = []
    politicians: list[Politician] = []

    def get_url_content(self):
        return requests.get(self.url).text

    def _crawler_1(self, soup: BeautifulSoup):
        for post in soup.findAll("div", {"id": "afdkarte"}):
            text = normalize(post.find("h5").text)
            h6 = normalize(post.find("h6").text).split()
            speaker = " ".join(h6[0:2])
            party = h6[2]
            self.speeches.append(Speech(text=text, speaker=Politician(name=speaker, party=party), url=self.url).save())

    def _crawler_2(self, soup: BeautifulSoup):
        for post in soup.findAll("ul", {"class": "bt-liste"}):
            for line in post.findAll("li"):
                text = normalize(line.text).split()
                text = remove_string(text, ["Prof", "Dr", "hc", "seit"])
                year = []
                for index, item in enumerate(text):
                    if item.isdigit():
                        year.append(item)
                        text = remove_string(text, [item])
                party = text[-1]
                text = remove_string(text, [party])
                self.politicians.append(Politician(name=" ".join(text), party=party).save())

    def _crawler_3(self, soup: BeautifulSoup):
        for post in soup.findAll("ul"):
            for line in post.findAll("li"):
                for element in line.findAll("a"):
                    # workaround: write to txt, manually normalization, read normalized txt
                    write_to_txt(element.text, "data/Regierungsmitglieder.txt", "w")

        text = read_from_txt("data/Regierungsmitglieder_normalisiert.txt")
        for item in text:
            politician = item[0:2]
            item = remove_string(item, politician)
            results = Politician.find({"name": {"$in": [" ".join(politician)]}})
            if Politician.find({"name": {"$in": politician}}):
                print(item)
            # TODO check if new politicians exists in db
            # self.politicians.append(Politician(name=" ".join(politician), party=" ".join(item)))

    def get_card_content(self):
        content = self.get_url_content()
        # übergebe html an beautifulsoup parser
        soup = BeautifulSoup(content, "html.parser")
        match self.index:
            case 1:
                self._crawler_1(soup)
            case 2:
                self._crawler_2(soup)
            case 3:
                self._crawler_3(soup)
