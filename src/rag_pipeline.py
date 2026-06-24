from transformers import pipeline

from src.retriever import retrieve_documents


PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust Financial.

Your task is to answer questions about customer complaints.

Use ONLY the provided context.

If the answer is not contained in the context,
say that you do not have enough information.

Instructions:
- Summarize recurring customer issues.
- Do not copy the complaints verbatim.
- Use professional business language.
- If multiple complaints mention similar issues, group them into themes.
- If the context is insufficient, say so.
- Keep the answer concise (3-6 bullet points).

Context:
{context}

Question:
{question}

Answer:
"""


class RAGPipeline:

    def __init__(
        self,
        llm_model="Qwen/Qwen2.5-0.5B-Instruct",
        k=5
    ):

        self.k = k

        self.generator = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=llm_model,
            device_map="cpu"
        )

    def retrieve(self, question):

        return retrieve_documents(
            question,
            k=self.k
        )

    def build_prompt(self, question, docs):

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        return PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

    def generate(self, prompt):

        output = self.generator(
            prompt,
            # max_new_tokens=200,
            do_sample=False,
            return_full_text=False
        )

        return output[0]["generated_text"].strip()
   

    def run(self, question):

        docs = self.retrieve(question)

        prompt = self.build_prompt(
            question,
            docs
        )

        answer = self.generate(prompt)

        return {
            "answer": answer,
            "sources": docs
        }