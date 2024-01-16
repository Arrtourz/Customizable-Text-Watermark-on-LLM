import openai
import logging
from tqdm import tqdm
from time import sleep

class gpt_logp:
    """ Language Model. """

    def __init__(self, api_key: str, model: str, sleep_time: int = 10):
        """ Language Model.

        @param api_key: OpenAI API key.
        @param model: OpenAI model.
        """
        logging.info(f'Loading Model: `{model}`')
        openai.api_key = api_key
        self.model = model
        self.sleep_time = sleep_time

    def get_logprobs(self, input_texts: str or list, *args, **kwargs):
        """ Compute the log probabilities and return corresponding tokens on recurrent LM.

        :param input_texts: A string or list of input texts for the encoder.
        :return: A list of tuples, where each tuple contains the log probabilities and corresponding token for a single input text.
        """
        single_input = type(input_texts) == str
        input_texts = [input_texts] if single_input else input_texts
        all_logprobs = []
        for text in tqdm(input_texts):
            while True:
                try:
                    completion = openai.Completion.create(
                        model=self.model,
                        prompt=text,
                        logprobs=5,
                        max_tokens=0,
                        temperature=0,
                        echo=True
                    )
                    break
                except Exception:
                    if self.sleep_time is None or self.sleep_time == 0:
                        logging.exception('OpenAI internal error')
                        exit()
                    logging.info(f'Rate limit exceeded. Waiting for {self.sleep_time} seconds.')
                    sleep(self.sleep_time)
            logprobs = completion['choices'][0]['logprobs']['token_logprobs']
            all_logprobs.append(logprobs)
        return all_logprobs,completion

