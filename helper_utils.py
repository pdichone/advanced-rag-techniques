# helper_utils.py
import numpy as np
import chromadb
import pandas as pd
from pypdf import PdfReader
import numpy as np


def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings


def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])


def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
    file_path (str): The path to the PDF file.

    Returns:
    str: The extracted text.
    """
    text = []
    with open(file_path, "rb") as f:
        pdf = PdfReader(f)
        for page_num in range(pdf.get_num_pages()):
            page = pdf.get_page(page_num)
            text.append(page.extract_text())
    return "\n".join(text)


def load_chroma(filename, collection_name, embedding_function):
    """
    Loads a document from a PDF, extracts text, generates embeddings, and stores it in a Chroma collection.

    Args:
    filename (str): The path to the PDF file.
    collection_name (str): The name of the Chroma collection.
    embedding_function (callable): A function to generate embeddings.

    Returns:
    chroma.Collection: The Chroma collection with the document embeddings.
    """
    # Extract text from the PDF
    text = extract_text_from_pdf(filename)

    # Split text into paragraphs or chunks
    paragraphs = text.split("\n\n")

    # Generate embeddings for each chunk
    embeddings = [embedding_function(paragraph) for paragraph in paragraphs]

    # Create a DataFrame to store text and embeddings
    data = {"text": paragraphs, "embeddings": embeddings}
    df = pd.DataFrame(data)

    # Create or load the Chroma collection

    collection = chromadb.Client().create_collection(collection_name)

    # Add the data to the Chroma collection
    for ids, row in df.iterrows():

        collection.add(ids=ids, documents=row["text"], embeddings=row["embeddings"])
        # collection.add(text=row["text"], embedding=row["embeddings"])

    return collection
