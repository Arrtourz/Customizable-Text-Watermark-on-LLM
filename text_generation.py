import openai

def generate_text_without_pattern(api_key, prompt, temperature, max_length):
    """
    Generate text using OpenAI GPT-3 model without a specific pattern.

    :param api_key: The API key for OpenAI.
    :param prompt: The prompt to send to the model.
    :param temperature: The temperature setting for the model.
    :return: Generated text from the model.
    """
    openai.api_key = api_key
    response = openai.Completion.create(
            engine="davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
    model_gen_text = response['choices'][0]['text']
    return model_gen_text


# gpt-3.5-turbo-instruct,
def generate_text_with_pattern(api_key, prompt, pattern, temperature, max_retries=3):
    openai.api_key = api_key

    pattern_index = 0
    model_gen_text = ""

    for _ in range(len(pattern)):
        retries = 0
        while retries < max_retries:
            try:
                response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt + model_gen_text,
                    max_tokens=1,
                    logprobs=5,
                    temperature=temperature,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                if response['choices'][0]['logprobs']['top_logprobs']:
                    top_logprobs = response['choices'][0]['logprobs']['top_logprobs'][0]
                    if top_logprobs:
                        next_token_rank = pattern[pattern_index % len(pattern)] - 1
                        next_token = sorted(top_logprobs.items(), key=lambda x: x[1], reverse=True)[next_token_rank][0]

                        if next_token == "<|endoftext|>":
                            model_gen_text += "."  # 插入默认的 token
                            logging.warning(" in response, inserting default token.")
                        else:
                            model_gen_text += next_token
                            break 

                    else:
                        logging.warning("Empty top_logprobs, retrying.")
                else:
                    logging.warning("No top_logprobs in response, retrying.")

                retries += 1

            except Exception as e:
                logging.exception(f"Exception occurred: {e}")
                retries += 1

        if retries == max_retries:
            logging.warning("Max tries limit")
            model_gen_text += "."

        pattern_index += 1

    return model_gen_text
