from langchain_text_splitters import RecursiveCharacterTextSplitter
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_text(text)