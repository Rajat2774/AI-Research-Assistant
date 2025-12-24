from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
import json
import re

from pydantic import BaseModel, Field
from typing import List, Dict

load_dotenv()

class CriticOutput(BaseModel):
    evaluation_scores: Dict[str, float] = Field(
        description="Numerical scores for quality, completeness, and rigor"
    )
    missing_topics: List[str] = Field(
        description="Important topics missing from the research"
    )
    redundant_sections: List[str] = Field(
        description="Sections that overlap or repeat content"
    )
    sections_needing_more_depth: List[str] = Field(
        description="Sections that require deeper analysis"
    )
    suggested_followup_questions: List[str] = Field(
        description="Concrete research questions to improve coverage"
    )


# 🔹 STEP 2: Critic function
def critique_research(research_question: str, summaries: dict):

    output_parser = JsonOutputParser(pydantic_object=CriticOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a strict academic research reviewer.

Return ONLY structured JSON that follows the provided schema.
Do NOT include explanations, comments, or extra text.
If something is not applicable, return an empty list.
All list items must be concise labels, not full sentences.

"""
        ),
        (
            "human",
            """
Research Question:
{research_question}

Research Summaries:
{summaries}

Evaluate the research using these criteria:
- Overall quality
- Missing important topics
- Redundant sections
- Sections needing more depth
- Suggested follow-up research questions

{format_instructions}
"""
        )
    ])

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2
    )

    chain = prompt | llm | output_parser

    try:
        critique = chain.invoke({
            "research_question": research_question,
            "summaries": summaries,
            "format_instructions": output_parser.get_format_instructions()
        })
    except OutputParserException as e:
        raw_text = e.llm_output

        # 🔧 SIMPLE JSON REPAIR
        repaired = raw_text.replace(" and ", ", ")

        # Remove trailing text outside JSON if any
        repaired = re.search(r"\{.*\}", repaired, re.DOTALL).group()

        critique = json.loads(repaired)

    
    return critique
