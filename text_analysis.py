import operator

def calculate_ppl(log_probs):
    """ Calculate the perplexity of a sequence of log probabilities.

    :param log_probs: List of log probabilities for each token in the text, 
                      or a list of lists of log probabilities.
    :return: The perplexity of the text.
    """
    if log_probs and isinstance(log_probs[0], list):
        flat_log_probs = [prob for sublist in log_probs for prob in sublist]
    else:
        flat_log_probs = log_probs

    N = len(flat_log_probs)
    if N > 0:
        return math.exp(-sum(flat_log_probs) / N)
    else:
        return float('inf')
        
def analyze_text_watermark(input_text, api_key):
    scorer_gpt35 = gpt_logp(api_key=api_key, model="davinci-002")

    logprobs, response_1 = scorer_gpt35.get_logprobs(input_text)

    samples_preview = response_1['choices'][0]['logprobs']['token_logprobs']

    gen_candidates_1 = response_1.choices[0].logprobs.top_logprobs
    gen_candidates_1 = [item for item in gen_candidates_1 if item is not None]

    sorted_gen_candidates_1 = [dict(sorted(item.items(), key=operator.itemgetter(1), reverse=True)) for item in gen_candidates_1]
    gen_candidates_format_1 = [[k for k in d.keys()] for d in sorted_gen_candidates_1]
    tokens_1 = response_1['choices'][0]['logprobs']['tokens']
    tokens_1 = tokens_1[1:]

    rank_output = [gen_candidates_format_1[i].index(tokens_1[i]) + 1 if tokens_1[i] in gen_candidates_format_1[i] else None for i in range(len(tokens_1))]
    rank_outputs = [3 if i is None else i for i in rank_output]

    return rank_outputs, samples_preview
