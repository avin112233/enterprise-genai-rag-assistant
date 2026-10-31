
import csv
from pathlib import Path

from llm import generate_answer

TEST_CASES = [
    {
        "question": "What validations are performed for Contact BP?",
        "expected_doc": 1,
        "expected_keywords": ["non-null", "boolean", "ratio"],
    },
    {
        "question": "Explain Hexagon scoring.",
        "expected_doc": 2,
        "expected_keywords": ["activity", "probabilit", "weight"],
    },
    {
        "question": "How do I troubleshoot a QA failure?",
        "expected_doc": 3,
        "expected_keywords": ["organization", "SQL", "schema"],
    },
]

results = []

for index, test in enumerate(TEST_CASES, start=1):
    print(f"\nRunning evaluation {index}/{len(TEST_CASES)}")
    print("Question:", test["question"])

    response = generate_answer(test["question"])
    answer = response["answer"] or ""

    retrieved_ids = [
        doc["metadata"]["chunk_id"]
        for doc in response["sources"]
    ]

    top_doc = retrieved_ids[0] if retrieved_ids else None

    retrieval_pass = top_doc == test["expected_doc"]

    keyword_matches = [
        keyword.lower() in answer.lower()
        for keyword in test["expected_keywords"]
    ]

    correctness_pass = all(keyword_matches)

    result = {
        "question": test["question"],
        "expected_doc": test["expected_doc"],
        "retrieved_top_doc": top_doc,
        "retrieval_pass": retrieval_pass,
        "keyword_check_pass": correctness_pass,
        "latency_seconds": response["latency"],
        "input_tokens": response["input_tokens"],
        "output_tokens": response["output_tokens"],
        "generated_answer": answer,
    }

    results.append(result)

    print("Retrieval passed:", retrieval_pass)
    print("Keyword check passed:", correctness_pass)
    print("Latency:", response["latency"], "seconds")

Path("results").mkdir(exist_ok=True)

output_file = Path("results/evaluation_results.csv")

with output_file.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

retrieval_accuracy = (
    sum(r["retrieval_pass"] for r in results) / len(results) * 100
)

keyword_pass_rate = (
    sum(r["keyword_check_pass"] for r in results) / len(results) * 100
)

average_latency = (
    sum(r["latency_seconds"] for r in results) / len(results)
)

print("\n" + "=" * 50)
print("RAG EVALUATION SUMMARY")
print("=" * 50)
print(f"Top-1 Retrieval Accuracy: {retrieval_accuracy:.2f}%")
print(f"Keyword Check Pass Rate: {keyword_pass_rate:.2f}%")
print(f"Average Latency: {average_latency:.2f} seconds")
print(f"Results saved to: {output_file}")
