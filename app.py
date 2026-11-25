import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "docs"
CHAT_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
NOT_FOUND_RESPONSE = "Not found in the document."

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict question-answering assistant. "
            "Answer the user's question using ONLY the provided context. "
            "Respond in one or two sentences. "
            'If the answer is absent from the context, reply with exactly: "Not found in the document."',
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int


def _extract_text(content: object) -> str:
    """Normalize LangChain AIMessage content to a plain string."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "".join(parts).strip()
    return str(content).strip()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Add it to your .env file or environment."
        )

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    app.state.vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    app.state.llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    yield


app = FastAPI(title="Gemini RAG API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    vectorstore: Chroma = app.state.vectorstore
    llm: ChatGoogleGenerativeAI = app.state.llm

    start = time.perf_counter()

    docs = vectorstore.similarity_search(question, k=3)
    sources = [doc.page_content for doc in docs]

    if not sources:
        latency_ms = round((time.perf_counter() - start) * 1000)
        return AskResponse(
            answer=NOT_FOUND_RESPONSE,
            sources=[],
            latency_ms=latency_ms,
        )

    context = "\n\n---\n\n".join(sources)
    messages = PROMPT_TEMPLATE.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    answer = _extract_text(response.content) or NOT_FOUND_RESPONSE

    latency_ms = round((time.perf_counter() - start) * 1000)
    return AskResponse(answer=answer, sources=sources, latency_ms=latency_ms)
