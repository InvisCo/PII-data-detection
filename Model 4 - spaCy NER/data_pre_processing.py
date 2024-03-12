import json
import random
from pathlib import Path

import spacy
from spacy.tokens import DocBin

INPUT_FOLDER = Path(__file__).parent.parent / "raw_data"
OUTPUT_FOLDER = Path(__file__).parent / "data"


def process_train_data(input_file: Path, output_folder: Path, split_ratio: float = 0.8):
    # Create a blank model for the English language
    nlp = spacy.blank("en")
    # Create DocBin objects to store the processed documents
    train_bin = DocBin()
    valid_bin = DocBin()

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    # Limit the number of documents for testing
    data = data[:100]

    # Split the data into training and validation sets
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    valid_data = data[split_index:]

    # Process each document and add it to the appropriate DocBin
    for data, doc_bin in [(train_data, train_bin), (valid_data, valid_bin)]:
        for document in data:
            text = ""
            current_pos = 0
            entities = []
            current_entity = None

            # Rebuild the text from tokens and whitespace
            for token, has_space, label in zip(
                document["tokens"], document["trailing_whitespace"], document["labels"]
            ):
                # Calculate the start and end positions of the token
                start = current_pos
                end = start + len(token)
                text += token
                current_pos += len(token)

                if has_space:
                    text += " "
                    current_pos += 1

                # Check label to update entity annotations
                if label.startswith("B-"):
                    # Save the previous entity if any
                    if current_entity:
                        entities.append(current_entity)

                    # Extract label type without BIO prefix
                    label_type = label.split("-", 1)[1]
                    current_entity = (start, end, label_type)

                elif label.startswith("I-") and current_entity:
                    # Update the end of the current entity if label is continuation
                    _, _, label_type = current_entity
                    current_entity = (current_entity[0], end, label_type)

                else:
                    # Save the previous entity if it's the end of an entity
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            # Ensure the last entity is added if the document ends with an entity
            if current_entity:
                entities.append(current_entity)

            # Create a Doc object and set its entities
            doc = nlp.make_doc(text)
            ents = [
                doc.char_span(start, end, label=label)
                for start, end, label in entities
                if doc.char_span(start, end, label=label) is not None
            ]
            doc.ents = ents

            # Add processed document to DocBin
            doc_bin.add(doc)

    # Save the DocBins to disk
    train_bin.to_disk(output_folder / "train.spacy")
    valid_bin.to_disk(output_folder / "valid.spacy")

    print(
        f"Processed {len(train_bin)} training and {len(valid_bin)} validation documents."
    )


def process_test_data(input_file: Path, output_file: Path):
    # Create a blank model for the English language
    nlp = spacy.blank("en")
    doc_bin = DocBin()

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for document in data:
        # Create a Doc object from the complete text and add it to the DocBin
        doc_bin.add(nlp.make_doc(document["full_text"]))

    # Save the DocBin to disk
    doc_bin.to_disk(output_file)

    print(f"Processed {len(doc_bin)} documents.")


if __name__ == "__main__":
    random.seed(546)
    process_train_data(INPUT_FOLDER / "train.json", OUTPUT_FOLDER)
    process_test_data(INPUT_FOLDER / "test.json", OUTPUT_FOLDER / "test.spacy")
