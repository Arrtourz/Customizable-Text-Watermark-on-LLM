from gpt_logp import gpt_logp
from signal_processing import generate_qpsk_signal, decode_to_ascii
from text_generation import generate_text_with_pattern
from text_analysis import analyze_text_watermark

encoding = tiktoken.get_encoding("p50k_base")

api_key = "YOUR-OPENAI-API"

prompt = "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. "
real_completion = "He will be teaching a beginner level class for everyone who wants to get better with their culinary skills. He will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information. The cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared."
temperature = 0

binary_data_list = []  # Ensure this list is initialized before the loop


# test for A-Z
for char in range(ord('A'), ord('Z') + 1):
    binary_data = format(char, '08b')
    binary_data_list.append(binary_data)

# test for a-z
for char in range(ord('a'), ord('z') + 1):
    binary_data = format(char, '08b')
    binary_data_list.append(binary_data)
    
# test for 0 to 9
for num in range(0, 10):
    binary_data = format(num, '08b')
    binary_data_list.append(binary_data)
    
    
print(binary_data_list)
test_res = []
pre = np.ones(40, dtype=int)
temperature=0.1
sample_n=20

for i in binary_data_list:
    sampled_qpsk_signal = generate_qpsk_signal(i)
    pattern = np.concatenate([pre, sampled_qpsk_signal])
    model_gen_text_W = generate_text_with_pattern(api_key, prompt, pattern, temperature)
    print(model_gen_text_W)
    rank_outputs_W, samples_preview_W = analyze_text_watermark(model_gen_text_W, api_key)
    print(len(rank_outputs_W))
    offset = 120-len(rank_outputs_W)
    ascii_character = decode_to_ascii(rank_outputs_W[40:] + offset*[3], sample_n)
    test_res.append(ascii_character)
    print("Decoded ASCII Character:", ascii_character)

    


binary_data_list = []  # Ensure this list is initialized before the loop

# 
for char in ['G','P','T','3']:
# for char in ['A', 'D', 'M', 'I', 'N']:
    ascii_value = ord(char)  # Convert character to ASCII value
    binary_data = format(ascii_value, '08b')  # Convert ASCII value to binary
    binary_data_list.append(binary_data)
    
print(binary_data_list)
test_res = []
# pre = np.ones(40, dtype=int)
temperature=0
sample_n=20
prompt = "Beginners BBQ Class Taking Place in Missoula! Do you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. Continue..."
model_gen_text_W=''
NW=''
for i in binary_data_list:
    sampled_qpsk_signal = generate_qpsk_signal(i)
#     pattern = np.concatenate([pre, sampled_qpsk_signal])
    pattern = sampled_qpsk_signal
    model_gen_text_W = generate_text_with_pattern(api_key, prompt, pattern, temperature)
    print(model_gen_text_W)
    prompt = prompt + model_gen_text_W
    NW = NW + model_gen_text_W
    rank_outputs_W, samples_preview_W = analyze_text_watermark(prompt, api_key)
    print(len(rank_outputs_W))
    offset = 80-len(rank_outputs_W)
#     ascii_character = decode_to_ascii(rank_outputs_W[40:] + offset*[3], sample_n)
#     print(len(rank_outputs_W[40:] + offset*[3]))
    ascii_character = decode_to_ascii(rank_outputs_W[-80:], sample_n)
    print(len(rank_outputs_W[-80:]))
    test_res.append(ascii_character)
    print("Decoded ASCII Character:", ascii_character)
    
    
