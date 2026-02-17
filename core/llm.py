import re

class LLM:
    def __init__(self, providers: dict):
        self.providers = providers

    def call(self, agent: str, prompt: str) -> str:
        return self.providers[agent].generate_content(prompt)

    def extract_code(self, response: str) -> str:
        blocks = re.findall(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
        return blocks[0] if blocks else response.strip()
