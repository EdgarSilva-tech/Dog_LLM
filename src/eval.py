from trulens.apps.langchain import TruChain
from trulens.core import TruSession
from LLM import chain, db, prompt, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from operator import itemgetter
from model import predict
from PIL import Image
import numpy as np
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from trulens.dashboard.display import get_feedback_result
from trulens.apps.langchain import WithFeedbackFilterDocuments
from trulens.dashboard import run_dashboard

session = TruSession()
session.reset_database()

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
context = TruChain.select_context(chain)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on(context.collect())  # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Feedback(
    provider.relevance_with_cot_reasons, name="Answer Relevance"
).on_input_output()
# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on(context)
    .aggregate(np.mean)
)

tru_recorder = TruChain(
    chain,
    app_name="ChatApplication",
    app_version="Chain1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

with tru_recorder as recording:
    breed = Image.open(r"C:\Users\edgar\OneDrive\Ambiente de Trabalho\AI_Projects\Dog_LLM\data\train\Samoyed\n02111889_108.jpg")
    llm_response = chain.invoke({"question": "How much do they shed", "breed": breed})

print(llm_response)

print(session.get_leaderboard())

last_record = recording.records[-1]
print(get_feedback_result(last_record, "Context Relevance"))

rec = recording.get()

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

# note: feedback function used for guardrail must only return a score, not also reasons
f_context_relevance_score = Feedback(provider.context_relevance)

filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
    retriever=db.as_retriever(), feedback=f_context_relevance_score, threshold=0.75
)

rag_chain = (
    RunnableParallel(
        {
            "context": itemgetter("question") | filtered_retriever,
            "breed": lambda inputs: predict(inputs["breed"]),
            "question": itemgetter("question"),
        }
    )
    | prompt
    | llm
    | StrOutputParser()
)

tru_recorder = TruChain(
    rag_chain,
    app_name="Dog_LLM",
    app_version="V1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

with tru_recorder as recording:
    llm_response = rag_chain.invoke({"question": "How much do they shed?", "breed": breed})

print(llm_response)

last_record = recording.records[-1]
print(get_feedback_result(last_record, "Context Relevance"))

rec = recording.get()

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

print(run_dashboard(session))

def eval_answer(chain, question, name, breed):

    tru_recorder = TruChain(
    chain,
    app_name=name,
    app_version="V1",
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness],
)

    with tru_recorder as recording:
        llm_response = chain.invoke({"question": question, "breed": breed})


    rec = recording.get()

    for feedback, feedback_result in rec.wait_for_feedback_results().items():
        return llm_response, feedback.name, feedback_result.result
