import chainlit as cl
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from utils.conversational_chain import LLMHandler
import config.chain_config as cfg
import os

# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize the LLM
if cfg.model_name.startswith("llama"):
    groq_api_key = os.getenv('GROQ_API_KEY')
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=cfg.model_name, temperature=cfg.temperature)
    embeddings = FastEmbedEmbeddings()
elif cfg.model_name.startswith("gpt"):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model_name=cfg.model_name, temperature=cfg.temperature, api_key=api_key)
    embeddings = OpenAIEmbeddings(model=cfg.embeddings_model)

welcome_message = """To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

@cl.on_chat_start
async def start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content=welcome_message, 
            accept=["text/plain", "application/pdf", "application/doc", "application/pptx"],
            max_size_mb=20,
            timeout=180,
        ).send()

    
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    conversation_handler = await cl.make_async(LLMHandler)(llm, file.name, file.path, embeddings)
    chain = conversation_handler.create_chain()

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("convhandle", conversation_handler)
    cl.user_session.set("chains", {file.name:chain})

@cl.on_message
async def main(message: cl.Message):
    convhandle = cl.user_session.get("convhandle")
    chains = cl.user_session.get("chains")  # type: RetrievalChain
    # cb = cl.AsyncLangchainCallbackHandler()

    if message.elements:
        for file in message.elements:
            if isinstance(file, cl.File):
                convhandle = await cl.make_async(LLMHandler)(llm, file.name, file.path, embeddings)
                chain = convhandle.create_chain()
                    
                # Store the chain in the user session
                chains[file.name] = chain

                res =  await chain.ainvoke({"input":message.content, "chat_history": convhandle.chat_history})
            else:
                await cl.Message(f"Unsupported file type: {file.type} for file: {file.name}").send()

    else:
        chain = list(chains.values())[-1]
        res = await chain.ainvoke({"input":message.content, "chat_history": convhandle.chat_history})

    answer = res["answer"]
    source_documents = res["context"] 

    convhandle.chat_history.extend(
        [
            HumanMessage(content=message.content),
            AIMessage(answer),
        ]
    )

    # text_elements = []  

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

    #     if source_names:
    #         answer += f"\nSources: {', '.join(source_names)}"
    #     else:
    #         answer += "\nNo sources found"

    await cl.Message(content=answer).send()




