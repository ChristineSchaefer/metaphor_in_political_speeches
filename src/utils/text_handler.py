import os

from src.config import BASE_DIR


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
