def tokenize(sentences, language_model) -> list:
    tokenized_texts = []
    for sent in sentences:
        tokenized_sent = []
        doc = language_model(sent)
        for token in doc:
            if not token.text.isspace():
                tokenized_sent.append(token.text.lower())
        tokenized_texts.append(tokenized_sent)
    return tokenized_texts
