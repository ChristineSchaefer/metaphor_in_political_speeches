import csv
import os

from src.config import BASE_DIR


def normalize(text: str):
    """
        This function removes certain special characters from the input string.

        @param text: string to normalize

        @returns normalized string
    """
    for char in '()"=+-.:,?\n':
        text = text.replace(char, " ")
    return text


def remove_string(token: list, to_remove: list):
    """
        Removes specific elements from a list.

        @param token: list of token
        @param to_remove: list of elements to remove

        @returns updated list
    """
    for i, t in enumerate(token):
        if t in to_remove:
            token[i] = ""
    return [x for x in token if x]


def write_to_txt(text: str, name: str, mode: str):
    """
       Writes a string to a text file.

       @param text: string
       @param name: name of file
       @param mode: read or write
    """
    file = open(os.path.join(BASE_DIR, name), mode, encoding="utf-8")
    file.write(text + "\n")
    file.close()


def write_list_with_dict_to_txt(data: list[dict], name: str, mode: str):
    """
        Writes a list of dictionaries to a text file.
        Each dictionary is written line by line, with each key value pair on a new line.

        @param data: list of dictionaries
        @param name: name of file
        @param mode: read or write
    """
    file = open(os.path.join(BASE_DIR, name), mode, encoding="utf-8")
    for element in data:
        for key, value in element.items():
            file.write('%s: %s\n' % (key, value))
        file.write("\n")
    file.close()


def read_from_txt(path: str) -> list:
    """
        Reads the contents of a text file, normalizes it and tokenizes it into list of words.

        @param path: path to file

        @returns content of file as list
    """
    content = []
    with open(os.path.join(BASE_DIR, path), encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            content.append(normalize(line).split())
    return content


def read_from_csv_and_write_to_dict(path: str) -> list[dict]:
    """
        Reads a csv file and writes the contents into a list of dictionaries.

        @param path: path to file

        @returns content of file as list with dict
    """
    with open(os.path.join(BASE_DIR, path), 'r', encoding="utf-8") as f:
        dict_reader = csv.DictReader(f, delimiter=',', fieldnames=[
            "_id",
            "speech_id",
            "sentence",
            "sentence_index",
            "verb",
            "lexem",
            "index_verb"
        ])

        return list(dict_reader)


def make_json_from_csv(csv_file: str):
    data = []

    with open(csv_file, encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            data.append(row)
    return data
