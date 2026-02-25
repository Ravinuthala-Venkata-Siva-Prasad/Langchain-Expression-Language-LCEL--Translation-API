from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Create Groq Model (API key auto-detected from environment)
model = ChatGroq(
    model= "openai/gpt-oss-120b"   # Safe & free Groq model
)

# Create Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following text into {language}."),
    ("user", "{text}")
])

# Output Parser
parser = StrOutputParser()

# Create LCEL Chain
chain = prompt | model | parser

# Create FastAPI app
app = FastAPI(
    title="LangChain Groq Server",
    version="1.0",
    description="Simple LCEL API using Groq"
)

# Add LangServe routes
add_routes(
    app,
    chain,
    path="/chain/playground/"
)

# Run Server
if __name__ == "__main__":
    uvicorn.run(app, host="http://127.0.0.1:8000/chain/playground/", port=8000)
