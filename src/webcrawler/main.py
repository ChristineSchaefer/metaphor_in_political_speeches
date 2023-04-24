from pymongo import MongoClient

from src.config import get_settings
from src.webcrawler.controller.crawler import CrawlerController

env = get_settings()
DB_CLIENT = MongoClient(env.mdb_connection_string(), uuidRepresentation="standard")
political_speeches_local = DB_CLIENT[env.db_name]

url = "https://goslar-gegen-rechtsextremismus.de/html/afd-sprueche.php"

if __name__ == "__main__":
    cc = CrawlerController(url=url)
    cc.get_card_content()
