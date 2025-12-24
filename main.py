# main.py

from planner import generate_sub_questions
from executor import research_sub_questions
from critic import critique_research

def run_research_agent(research_question: str):
    print("\n📌 Research Question:")
    print(research_question)

    print("\n🧠 Planning sub-questions...\n")
    sub_questions = generate_sub_questions(research_question)

    summaries = {}

    for idx, sub_q in enumerate(sub_questions, 1):
        print(f"🔍 [{idx}/{len(sub_questions)}] Researching:")
        print(sub_q)

        summary = research_sub_questions(sub_q)
        summaries[sub_q] = summary

    return summaries



if __name__ == "__main__":
    question = "What are recent methods for explainable AI in healthcare?"

    research_results = run_research_agent(question)

    print("\n📄 FINAL RESEARCH OUTPUT\n")

    for i, (q, summary) in enumerate(research_results.items(), 1):
        print(f"\n{i}. {q}")
        print(summary)

    print("\n🧠 CRITIC AGENT REVIEW\n")

    critique = critique_research(question, research_results)
    # print(type(critique))
    for key, value in critique.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, list):
            for item in value:
                print(f"- {item}")
        else:
            print(value)