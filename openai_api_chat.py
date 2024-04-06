import openai
import tiktoken
from config import openai_api_conf as conf
from rate_controller import RateController
from typing import Tuple, Iterable, AsyncIterable

if conf.OPENAI_API_BASE:
    openai.api_base = conf.OPENAI_API_BASE
openai.api_key = conf.OPENAI_API_KEY

rate_control = RateController(
    limit=conf.N_LIMIT,
    period_sec=conf.PERIOD_SEC,
    backoff_max_retry=conf.BACKOFF_MAX_RETRY,
    backoff_on_errors=(
        openai.error.RateLimitError,
        openai.error.APIError
    )
)


# wrap the api to apply rate control and retry logics
@rate_control.apply(asynchronous=False)
def chat_completion_create(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)


@rate_control.apply(asynchronous=True)
async def chat_completion_acreate(*args, **kwargs):
    return await openai.ChatCompletion.acreate(*args, **kwargs)


class OpenaiAPIChat:
    """
    A class for making conversation with Openai style REST api easier.
    It keeps chat history, takes care of streaming response,
    rate control and retry logics, with asynchronous support.
    """

    def __init__(
            self,
            model_name: str = conf.DEFAULT_MODEL_NAME,
            system_prompt: str = conf.DEFAULT_SYS_PROMPT,
            max_retry: int = 10
    ):
        """
         Initialize the chat instance.
        :param model_name: The name of the model to be used.
        :param system_prompt: The system prompt to be used.
        :param max_retry: The maximum number of retry attempts for API calls.
        """
        self.model_name = model_name
        self.chat_log = []
        self.sys_prompt = system_prompt
        self.encoding = tiktoken.get_encoding(conf.ENC_MAP.get(model_name, 'cl100k_base'))

        self.max_retry = max_retry
        self.retry_cnt = 0

    def _make_msg(self, user_prompt, to_continue=False):
        msg = [
            {
                'role': 'system',
                'content': self.sys_prompt
            },
            *self.chat_log
        ]
        if not to_continue:
            msg.append(
                {
                    'role': 'user',
                    'content': user_prompt
                }
            )
        return msg

    def clear(self):
        """Clear chat history"""
        self.chat_log = []

    def get_response(
            self,
            user_prompt: str,
            to_continue: bool = False,
            **extra_kwargs
    ) -> tuple[str, str]:
        """
        Get a response from the model synchronously.
        :param user_prompt: The user's input
        :param to_continue: Flag indicating if the conversation should continue
        :param extra_kwargs: Additional keyword arguments for API call
        :return: A tuple containing the response content and finish reason
        """
        retry_cnt = 0
        while retry_cnt < self.max_retry:
            try:
                response = chat_completion_create(
                    model=self.model_name,
                    messages=self._make_msg(user_prompt, to_continue),
                    stream=False,
                    **extra_kwargs
                )
                full_content = response.choices[0].message['content']
                finish_reason = response.choices[0].finish_reason
                self.chat_log += OpenaiAPIChat.round_format(user_prompt, full_content)
                return full_content, finish_reason
            except Exception as error:
                retry_cnt += 1
                print(error, f'retry: {retry_cnt} / {self.max_retry}')
            print('max retry reached')
            return '', ''

    async def get_aresponse(
            self,
            user_prompt: str,
            to_continue: bool = False,
            **extra_kwargs
    ) -> tuple[str, str]:
        """
        Get a response from the model asynchronously.
        :param user_prompt: The user's input
        :param to_continue: Flag indicating if the conversation should continue
        :param extra_kwargs: Additional keyword arguments for API call
        :return: A tuple containing the response content and finish reason
        """
        retry_cnt = 0
        while retry_cnt < self.max_retry:
            try:
                response = await chat_completion_acreate(
                    model=self.model_name,
                    messages=self._make_msg(user_prompt, to_continue),
                    stream=False,
                    **extra_kwargs
                )
                full_content = response.choices[0].message['content']
                finish_reason = response.choices[0].finish_reason
                self.chat_log += OpenaiAPIChat.round_format(user_prompt, full_content)
                return full_content, finish_reason

            except Exception as error:
                retry_cnt += 1
                print(error, f'retry: {retry_cnt} / {self.max_retry}')
            print('max retry reached')
            return '', ''

    def get_stream_response(
            self,
            user_prompt: str,
            to_continue: bool = False,
            **extra_kwargs
    ) -> Iterable[Tuple[str, str]]:
        """
        Get streaming response from the model synchronously.
        :param user_prompt: The user's input
        :param to_continue: Flag indicating if the conversation should continue
        :param extra_kwargs: Additional keyword arguments for API call
        :return Generator yields a tuple of content shreds and stop reason
        """
        response = chat_completion_create(
            model=self.model_name,
            messages=self._make_msg(user_prompt, to_continue),
            stream=True,
            **extra_kwargs
        )
        role = None
        full_content = ''
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            role = delta.get_root('role', role)
            content = delta.get_root('content', '')
            full_content += content
            finish_reason = chunk['choice'][0]['finish_reason']
            yield content, finish_reason
        self.chat_log.append({'role': 'user', 'content': user_prompt})
        self.chat_log.append({'role': role, 'content': full_content})

    async def get_stream_aresponse(
            self,
            user_prompt: str,
            to_continue: bool = False,
            **extra_kwargs
    ) -> AsyncIterable[Tuple[str, str]]:
        """
        Get streaming response from the model asynchronously.
        :param user_prompt: The user's input
        :param to_continue: Flag indicating if the conversation should continue
        :param extra_kwargs: Additional keyword arguments for API call
        :return AsyncGenerator yields a tuple of content shreds and stop reason
        """
        response = await chat_completion_acreate(
            model=self.model_name,
            messages=self._make_msg(user_prompt, to_continue),
            stream=True,
            **extra_kwargs
        )
        role = None
        full_content = ''
        async for chunk in response:
            delta = chunk['choices'][0]['delta']
            role = delta.get('role', role)
            content = delta.get('content', '')
            full_content += content
            finish_reason = chunk['choices'][0]['finish_reason']
            yield content, finish_reason
        self.chat_log.append({'role': 'user', 'content': user_prompt})
        self.chat_log.append({'role': role, 'content': full_content})

    def n_tokens(self, text):
        """
        Calculate the number of tokens for the model in a given text.
        :param text: The text to calculate token count.
        :return: Number of tokens
        """
        return len(self.encoding.encode(text))

    @staticmethod
    def round_format(user, assistant):
        return [
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': assistant}
        ]
