from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
import bs4


def text_loader():
    urls =["https://www.stuart-turner.co.uk/support/how-to/how-to-make-a-shower-pump-quieter",
           "https://www.domesticpumps.ie/questions/","https://www.stuart-turner.co.uk/support/faqs",
           "https://www.nrmplumbingandheating.ie/stuart-turner-pump-repair",
           "https://www.stuart-turner.co.uk/support/techassist"]
    loader = WebBaseLoader(
      web_paths=(urls),
      bs_kwargs=dict(
          parse_only=bs4.SoupStrainer(
              class_=("dropdown-title", "dropdown-content", "ewd-ufaq-faq-title-text",
                      "ewd-ufaq-faq-body","ewd-ufaq-faq-body ewd-ufaq-hidden","generic-content","post-content")
          )
      ),
  )
    docs = loader.load()
    dirloader = DirectoryLoader("/Users/navjeetkaur/Downloads/DissertationDocs", 
                                glob="**/*.*",use_multithreading=True,show_progress=True,loader_cls=PyPDFLoader)
    manuals = dirloader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(manuals)
    docs = text_splitter.split_documents(docs)
    documents.extend(docs)
    return documents