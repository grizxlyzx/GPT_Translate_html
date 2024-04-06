import os

# >>> openai api >>>
OPENAI_API_BASE = ''  # put your api base here, leave empty if you use service from Openai
OPENAI_API_KEY = ''  # put your api key here, or set it as OPENAI_API_KEY environment variable
DEFAULT_MODEL_NAME = 'gpt-4-1106-preview'
DEFAULT_SYS_PROMPT = 'you are a helpful assistant.'

ENC_MAP = {
    'gpt-4-0613': 'cl100k_base',
    'gpt-4-1106-preview': 'cl100k_base',
    'gpt-3.5-turbo': 'cl100k_base',
    'text-embedding-ada-002': 'cl100k_base',
    'text-davinci-002': 'p50k_base',
    'text-davinci-003': 'p50k_base',
    'davinci': 'p50k_base'
}
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', OPENAI_API_KEY)
# <<< openai api <<<

# >>> api rate control >>>
N_LIMIT = 1
PERIOD_SEC = 3.5
BACKOFF_MAX_RETRY = 10
# <<< api rate control <<<
