import json
import os
from typing import Mapping, Any

from dict2xml import dict2xml

from src.config import BASE_DIR
from src.data_handler.models.speech import Speech


def normalize(text: str):
    for char in '()"=+-.,\n':
        text = text.replace(char, " ")
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


def save_collection_in_separate_xml(search: dict, path: str):
    speeches = Speech.find(filter=search)
    print(len(speeches))

    size_of_the_split = 1
    total = len(speeches) // size_of_the_split
    print(total + 1)

    for i in range(total):
        speech = speeches[i].dict(exclude={"_id"})
        xml = dict2xml(speech, wrap='speech', indent="   ")
        file = open(
                      os.path.join(BASE_DIR, path) + "speeches_split_" + str(i + 1) + ".xml",
                      "w")
        file.write(xml)
        file.close()


if __name__ == "__main__":
    save_collection_in_separate_xml({"url": "Bundesregierung.xml"}, "data/collections/bundesregierung_split/")
