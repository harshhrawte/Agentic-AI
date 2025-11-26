from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import re

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

application_info = {
    "name": None,
    "email": None,
    "skills": None
}

def extract_name(text: str):
    pattern1 = re.search(r"(?:my name is|i am|this is)\s+([A-Za-z][A-Za-z\s]+)", text, re.IGNORECASE)
    if pattern1:
        return pattern1.group(1).strip()

    first_line = text.strip().split("\n")[0]
    if 2 <= len(first_line.split()) <= 4 and not re.search(r"(resume|cv|profile|summary)", first_line, re.IGNORECASE):
        return first_line.strip()

    pattern2 = re.search(r"^([A-Za-z\s]{3,40})\s*\n.*email", text, re.IGNORECASE | re.MULTILINE)
    if pattern2:
        name = pattern2.group(1).strip()
        if not re.search(r"(resume|curriculum|vitae)", name, re.IGNORECASE):
            return name

    pattern3 = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", text)
    if pattern3:
        return pattern3.group(1).strip()

    return None

def extract_application_info(text: str) -> str:
    name = extract_name(text)
    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)
    skills_match = re.search(r"(?:skills are|i know|i can use)\s+(.+)", text, re.IGNORECASE)

    response = []

    if name:
        application_info["name"] = name
        response.append("Name saved.")

    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("Email saved.")

    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("Skills saved.")

    if not response:
        return " I couldn't extract any info. Please provide your name, email, or skills."

    return " ".join(response) + " Let me check what else I need."

def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return (
            f" You're ready! "
            f"Name: {application_info['name']}, "
            f"Email: {application_info['email']}, "
            f"Skills: {application_info['skills']}."
        )
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f" Still need: {', '.join(missing)}."

tools = [
    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extract name, email, and skills from the user's text."
    ),
    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Check whether name, email, and skills are collected.",
        return_direct=True
    ),
]

SYSTEM_PROMPT = """
You are a helpful job application assistant.
Your goal is to collect the user's name, email, and skills.
Use the provided tools to extract information.
Once everything is collected, confirm completion and stop.
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": SYSTEM_PROMPT}
)

print("Hi! I'm your job application assistant. Please tell me your name, email, and skills.")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print(" Bye! Good luck.")
        break

    response = agent.invoke({"input": user_input})
    print("Bot:", response["output"])

    if "you're ready" in response["output"].lower():
        print(" Application info complete!")
        break
