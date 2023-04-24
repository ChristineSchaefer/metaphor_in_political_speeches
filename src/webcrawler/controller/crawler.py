import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

from src.webcrawler.models.speech import Speech


class CrawlerController(BaseModel):
    url: str
    content: list[Speech] = list

    def get_url_content(self):
        return requests.get(self.url).text

    def get_card_content(self):
        content = self.get_url_content()
        # übergebe html an beautifulsoup parser
        soup = BeautifulSoup(content, "html.parser")
        for post in soup.findAll('div', {'id': 'afdkarte'}):
            print(post.find('h5'))
            print(post.find('h6'))
