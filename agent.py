# code connects your agent/chatbot to Gemini 2.0 using an OpenAI-style interface, pulling the API key from your .env file.

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled
import os

load_dotenv()
set_tracing_disabled(True)

provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model= "gemini-2.0-flash-exp",
    openai_client = provider,
)

#web developer
web_developer = Agent(
    name ="Web Developer Expert",
    instructions="Build responsive and performant websites suing modern frameworks.",
    model = model,
    handoff_description="handoff to web developer if the task is related to web development."
)

#app developer
app_developer = Agent(
    name ="App Development Expert",
    instructions="Build mobile application using modern frameworks.",
    model=model,
    handoff_description="handoff to app developer if the task is related to mobile app development."
)

#marketing agent
marketing = Agent(
    name ="Marketing Expert",
    instructions="Create and execute marketing strategies to promote products and services.",
    model=model,
    handoff_description="handoff to marketing agent if the task is related to marketing"
)

#content writting agent

content_writer = Agent(
   name="Content Writer",
   instructions="Write engaging, SEO-optimized content for blogs, websites, and social platforms.",
   model=model,
   handoff_description="handoff to content writer if the task involves writing or generating content.",
)

#Graphic Designer Agent
graphic_designer = Agent(
   name="Graphic Designer",
   instructions="Provide assistance with graphic design tasks including UI/UX design, logo ideas, banners, and tool recommendations such as Figma, Adobe XD, or Canva.",
   model=model,
   handoff_description="handoff to graphic designer if the query is about design, UI/UX, logos, or visual creativity."
)

async def myAgent(user_input):
   manager = Agent(
    name="manager",
    instructions="You will chat with the user and  delegrate tasks to speacialized agents based on their request.",
    model=model,
    handoffs=[web_developer,app_developer,marketing,content_writer,graphic_designer]
   )

   response = await Runner.run(
       manager,
       input=user_input
) 

   return response.final_output


#code that will connect gemini api key and tell what model and provider we are using
# in this we'll make multi agents like web developer, mobile developer, marketing expert etc