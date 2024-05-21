# Translator
from transformers import pipeline

text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
translator = pipeline("translation_en_to_fr", model="my_awesome_opus_books_model")

print(translator(text))

# Summarizer
from transformers import pipeline

text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."

summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
summarizer(text)

print(summarizer(text))

# Question answering
from transformers import pipeline

question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
answer = question_answerer(question=question, context=context)

print(answer)