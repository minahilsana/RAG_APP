from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEXT_DIR = Path("data/texts")

def split():
    # Initialize the splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # max size of each chunk (in characters)
        chunk_overlap=200,      # overlap between chunks to preserve context
        length_function=len,    # use number of characters to measure length
        is_separator_regex=False  # treat separators as plain text, not regex
    )

    all_chunks = []

    for txt_file in TEXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")

        # Create LangChain Document objects with metadata
        docs = text_splitter.create_documents(
            [text],
            metadatas=[{"source": txt_file.name}]
        )

        all_chunks.extend(docs)

    print(f"📄 Created {len(all_chunks)} chunks from {len(list(TEXT_DIR.glob('*.txt')))} files")
    return all_chunks

# if __name__ == "__main__":
#     docs = load_and_split()
#     # print(docs[0]) 
#     # print(docs[5]) # show an example chunk
