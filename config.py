# OPENAI_API_KEY = "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 "

# "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"

import os

class OpenAIConfig:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
