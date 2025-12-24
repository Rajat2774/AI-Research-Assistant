from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def research_sub_questions(sub_question:str):

    search_tool = TavilySearch(max_results=5)

    search_results = search_tool.invoke({
        "query": sub_question
    })

    search_results = "\n".join(
        r["content"] for r in search_results if "content" in r
    )
    summary_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a research assistant. "
            "You must NEVER reveal your reasoning, analysis, or internal thoughts. "
            "Do NOT include <think> tags or explanations. "
            "Output ONLY the final research summary."
        ),
        (
            "human",
            """
    Research Question:
    {sub_question}

    Search Results:
    {search_results}

    Write a concise academic-style summary:
    - Focus on methods, trends, and limitations
    - 1–2 short paragraphs
    - Factual and neutral
    - 1–2 short paragraphs ONLY
    - Do NOT repeat sentences or ideas
    - If information is redundant, summarize it once
    - Stop writing once the summary is complete
    """
        )
    ])



    llm = ChatGroq(
        temperature=0.3,
        model="llama-3.1-8b-instant"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()

    summary = summary_chain.invoke({
        "sub_question": sub_question,
        "search_results": search_results
    })
    return summary

