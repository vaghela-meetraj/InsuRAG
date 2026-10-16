import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# 1. Load environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
jina_key = os.getenv("JINA_API_KEY")

# 2. Load documents
data_path = os.path.join("data", "policy.txt")
loader = TextLoader(data_path, encoding="utf-8")
documents = loader.load()

# 3. Chunk documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(documents)

# 4. Embeddings & Vector Store
embedding_model = JinaEmbeddings(
    session=None,
    model_name="jina-embeddings-v2-base-en"
)
vector_store = FAISS.from_documents(chunks, embedding_model)

# 5. Retriever & Tool definition
retreiver = vector_store.as_retriever(search_kwargs={"k": 8})

def search_insurance_policy(question: str) -> str:
    """Search the insurance policy files or documents regarding the questions."""
    matching_chunks = retreiver.invoke(question)
    return "\n\n".join(chunk.page_content for chunk in matching_chunks)

# 6. LLM Model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# 7. Agent
insurance_assistant = create_agent(
    model=llm,
    tools=[search_insurance_policy],
    system_prompt="""
        You are an assistant for answering questions about health insurance policies.
        Answer the question using only the provided context.
        If the answer is not contained in the context, say that you do not know.
        Keep your response clear, accurate, and concise without making assumptions.
        """
)

# 8. Function to ask questions
def ask_insurance_assistant(question: str) -> str:
    print(f"Question: {question}")
    response = insurance_assistant.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })
    answer = response["messages"][-1].content
    print(f"Answer: {answer}\n")
    return answer


if __name__ == "__main__":
    # Test query
    ask_insurance_assistant("Within how many hours must hospitalization be intimated to the insurer?")
