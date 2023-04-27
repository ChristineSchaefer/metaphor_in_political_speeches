import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.data_handler.models.politician import Politician
from src.data_handler.models.speech import Speech


def normalize(text: str):
    for char in '()"=+-.,\n':
        text = text.replace(char, "")
    return text


def remove_title(token: list, to_remove: list):
    for i, t in enumerate(token):
        if t in to_remove:
            token[i] = ""
    return [x for x in token if x]


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
            for l in post.findAll("li"):
                text = normalize(l.text).split()
                text = remove_title(text, ["Prof", "Dr", "hc"])
                print(text)
                # TODO split list into str and int

    def get_card_content(self):
        content = self.get_url_content()
        # übergebe html an beautifulsoup parser
        soup = BeautifulSoup(content, "html.parser")
        match self.index:
            case 1:
                self._crawler_1(soup)
            case 2:
                self._crawler_2(soup)
