# This is the example generation script.

import ollama
from openai import OpenAI
import os
import json
from jsonschema import validate, ValidationError
from dotenv import load_dotenv

# main system prompt. Schema hard-coded for JSON output.
# The keys required for SPIDER compatiblity are:
#   question
#   sql
# The remaining keys are useful for the generation pipeline,
# allowing for more analysis and specific examples
# and to indicate what difficulty and spread of SQL types
# there are.

SYSTEM_PROMPT = """
You are a data generation engine.

You MUST output valid JSON only.
Do NOT include markdown.
Do NOT include explanations.
Do NOT include extra keys.

The output MUST conform EXACTLY to this schema:

{
  "domain": string,
  "difficulty": "easy" | "medium" | "hard",
  "question": string,
  "sql": string,
  "tables_used": string[],
  "joins_used": string[],
  "has_subquery": boolean,
  "notes": string
}

Rules:
- SQL must be syntactically valid ANSI SQL
- Use only tables and columns defined in the provided schema
- Difficulty levels:
  - easy: single-table SELECT
  - medium: JOINs or GROUP BY
  - hard: subqueries or nested SELECTs
- Do NOT hallucinate tables or columns
"""

class LLMClient:
    def generate(self, system_prompt, user_prompt):
        raise NotImplementedError


class OllamaClient(LLMClient):
    def __init__(self, model="llama3"):
        self.model = model

    def generate(self, system_prompt, user_prompt):
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.9,
                     "top_p": 0.95,
                     "repeat_penalty": 1.2,
                     "num_ctx": 4096}
        )
        return response["message"]["content"]


class OpenAIClient(LLMClient):
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate(self, system_prompt, user_prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7
        )
        return response.choices[0].message.content


def get_client(model="qwen3:14b"):
    backend = os.getenv("LLM_BACKEND", "ollama")
    if backend == "openai":
        return OpenAIClient()
    return OllamaClient(model=model)


def build_prompt(domain_spec, difficulty, recent_examples):
    avoided_patterns = "\n".join(
        f"- {ex['sql']}" for ex in recent_examples[-5:]
    )

    return f"""
Domain: {domain_spec['name']}

Database schema:
{domain_spec['schema']}

Target difficulty: {difficulty}

Do NOT repeat SQL patterns similar to:
{avoided_patterns}

Generate ONE novel text-to-SQL example.
"""


JSON_SCHEMA = {
    "type": "object",
    "required": [
        "domain",
        "difficulty",
        "question",
        "sql",
        "tables_used",
        "joins_used",
        "has_subquery",
        "notes"
    ],
    "properties": {
        "domain": {"type": "string"},
        "difficulty": {
            "type": "string",
            "enum": ["easy", "medium", "hard"]
        },
        "question": {"type": "string"},
        "sql": {"type": "string"},
        "tables_used": {
            "type": "array",
            "items": {"type": "string"}
        },
        "joins_used": {
            "type": "array",
            "items": {"type": "string"}
        },
        "has_subquery": {"type": "boolean"},
        "notes": {"type": "string"}
    },
    "additionalProperties": False
}

def parse_and_validate(raw_output, retries=2):
    last_error = None

    for _ in range(retries + 1):
        try:
            obj = json.loads(raw_output)
            validate(instance=obj, schema=JSON_SCHEMA)
            return obj
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e

    raise ValueError(f"Invalid JSON after retries: {last_error}")


def write_jsonl(examples, path):
    with open(path, "a") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def generate_dataset(domain_spec, model="qwen3:14b", n_per_level=50):
    client = get_client(model=model)
    examples = []
    print("Generating examples.....\n")
    for difficulty in ["easy", "medium", "hard"]:
        print("Generating {} examples.....\n".format(difficulty))
        for i in range(n_per_level):
            prompt = build_prompt(domain_spec, difficulty, examples)
            raw = client.generate(SYSTEM_PROMPT, prompt)
            print("raw text: {}\n".format(raw))
            example = parse_and_validate(raw)
            print("parsed example: {}\n".format(example))
            examples.append(example)
            write_jsonl([example], os.path.join("data", "generated", domain_spec['name']))
            print(" {} Completed..\n".format(i + 1))

    return examples


if __name__ == "__main__":
    # Load the environment variables.
    load_dotenv()

    # These should be either generated or entered by user.
    DOMAIN = "university"

    # Example Domain Specification. This would be filled out by user of system.
    domain_spec = {
        "name": "university",
        "schema": """
    students(student_id, name, major)
    courses(course_id, title, department)
    enrollments(student_id, course_id, grade)
    """
    }

    examples = generate_dataset(domain_spec, n_per_level=50)