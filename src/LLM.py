from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from model import predict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

text_data = TextLoader("data/text_data/Breed_Data.txt", encoding="utf-8").load()
text_data = ''.join(doc.page_content for doc in text_data)
splitter = RecursiveCharacterTextSplitter(chunk_size=512,
    chunk_overlap=20)
splits = splitter.split_text(text_data)
docs = splitter.create_documents(splits)
embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma.from_documents(docs, embeddings_model)

system_prompt = (
    "Use the given context: {context} to answer questions about this dog breed: {breed}"
    "If you don't know the answer, say you don't know. "
    "question: {question}"
)

prompt = ChatPromptTemplate.from_template(system_prompt)

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.1)
chain = (
    RunnableParallel(
        {
            "context": itemgetter("question") | db.as_retriever(),
            "breed": lambda inputs: predict(inputs["breed"]),
            "question": itemgetter("question"),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)