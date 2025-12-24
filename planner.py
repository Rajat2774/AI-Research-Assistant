from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def generate_sub_questions(research_question:str):

    response_schemas = [
        ResponseSchema(
            name="sub_questions",
            description="A list of focused sub-questions for research"
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["research_question"],
        partial_variables={"format_instructions": format_instructions},
        template="""
    You are a research planning assistant.

    Your task is to break the given research question into 4 to 6 clear,
    non-overlapping sub-questions suitable for a literature review.

    Research Question:
    {research_question}

    {format_instructions}
    """
    )




    llm = ChatGroq(
        temperature=0.2,   # Low creativity = stable output
        model="llama-3.1-8b-instant"
    )

    chain = prompt | llm | StrOutputParser()

    raw_output = chain.invoke({
        "research_question": research_question
    })
    parsed_output = output_parser.parse(raw_output)
    return parsed_output["sub_questions"]




