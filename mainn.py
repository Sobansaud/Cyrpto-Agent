from agents import Agent , Runner ,AsyncOpenAI , OpenAIChatCompletionsModel , RunConfig
from dotenv import load_dotenv
import chainlit as cl
from tools import get_crypto
import os

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("Gemini APi Key is not defined")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai",
)

model = OpenAIChatCompletionsModel(
    model = "gemini-2.0-flash",
    openai_client=client,
)

config = RunConfig(
    model = model,
    model_provider=client,
    tracing_disabled=True
)

agent = Agent(
    name = "Crypto Price Agent",
    instructions= "You are a helpful Agent that gives real time cryptocurrency prices",
    tools = [get_crypto]
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history" , [])
    await cl.Message(content= " Welcome to Crypto Price Agent!").send()
    


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history")
    history.append({"role": "user", "content": message.content})

    print("User input history:", history)

    try:
        result = Runner.run_sync(
            agent,
            input=history,
            run_config=config
        )
        final = result.final_output
    except Exception as e:
        final = f"Error occurred: {str(e)}"
        
    await cl.Message(content=final).send()
    history.append({"role": "assistant", "content": final})
    cl.user_session.set("history", history)
