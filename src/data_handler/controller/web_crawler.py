import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from src.data_handler.models.politician import Politician
from src.data_handler.models.speech import Speech
from src.utils.text_handler import normalize, remove_string, write_to_txt, read_from_txt


class CrawlerController(BaseModel):
    """
        Controller class for managing a web crawler.
        This class is responsible for fetching and parsing web content from a specified URL
        and maintaining collections of speeches and politicians.
    """
    url: str = Field(..., description="url")
    index: int = Field(..., description="index of crawler")
    speeches: list[Speech] = Field(default_factory=list, description="list of speech objects")
    politicians: list[Politician] = Field(default_factory=list, description="list of politician objects")

    def get_url_content(self):
        """
            Fetches the content from the target URL.

            @return fetched page content as a string
        """
        return requests.get(self.url).text

    def _crawler_1(self, soup: BeautifulSoup):
        """
            Private method to crawl and parse content specific to HTML divs with id "afdkarte".
            Extracts the speech text, speaker, and party data and creates a new Speech object
            which is then saved in the speeches list.

            @param soup: BeautifulSoup object for parsing webpage
        """
        for post in soup.findAll("div", {"id": "afdkarte"}):
            text = normalize(post.find("h5").text)
            h6 = normalize(post.find("h6").text).split()
            speaker = " ".join(h6[0:2])
            party = h6[2]
            self.speeches.append(Speech(text=text, speaker=Politician(name=speaker, party=party), url=self.url).save())

    def _crawler_2(self, soup: BeautifulSoup):
        """
            Private method to crawl and parse content specific to HTML unordered lists with class "bt-liste".
            Extracts data about politicians and creates new Politician instances, which are then saved in the politicians list.

            @param soup: BeautifulSoup object for parsing webpage
        """
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
        """
            Private method to crawl and parse content specific to HTML unordered lists.
            Gathers data from list items to create and save new Politician instances.
            Uses a workaround to perform a manual normalization operation, and reads the normalized data from a text file.

            @param soup: BeautifulSoup object for parsing webpage
        """
        for post in soup.findAll("ul"):
            for line in post.findAll("li"):
                for element in line.findAll("a"):
                    # workaround: write to txt, manually normalization, read normalized txt
                    write_to_txt(element.text, "data/Regierungsmitglieder.txt", "w")

        text = read_from_txt("data/Regierungsmitglieder_normalisiert.txt")
        for item in text:
            politician = item[0:2]
            party = remove_string(item, politician)
            if len(Politician.find({"name": {"$in": [" ".join(politician)]}})) <= 0:
                self.politicians.append(Politician(name=" ".join(politician), party=" ".join(party)).save())

    def get_card_content(self):
        """
            Fetches content from the target URL and parses it into a BeautifulSoup object.
            According to the defined index, it delegates data extraction to the relevant crawler method.
        """
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
