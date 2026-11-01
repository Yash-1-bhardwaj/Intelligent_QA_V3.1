from pathlib import Path

from qa_pipeline import QAPipeline


BASE_DIR = Path(__file__).resolve().parent.parent


pipeline = QAPipeline(BASE_DIR)


result = pipeline.ask(
    "What are Query, Key and Value?"
)


print("\n==============================")
print("QUESTION")
print("==============================")

print(result["question"])


print("\n==============================")
print("ANSWER")
print("==============================")

print(result["answer"])


print("\n==============================")
print("PAGES")
print("==============================")

print(result["pages"])