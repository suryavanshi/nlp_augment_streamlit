import streamlit as st
import torch
from transformers import pipeline
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import random
import tokenizers

st.title("D-Labeler")


user_input = st.text_area("Enter Sentence", "I went to see a movie in the theater")

@st.cache(hash_funcs={tokenizers.Tokenizer: id})
def load_en2de():
    en2de = pipeline("translation_en_to_de", model='t5-base')
    return en2de

@st.cache(hash_funcs={tokenizers.Tokenizer: id},allow_output_mutation=True)
def load_de2en():
    mname = "facebook/wmt19-de-en"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model_de_to_en = FSMTForConditionalGeneration.from_pretrained(mname)

    return tokenizer, model_de_to_en

@st.cache(hash_funcs={tokenizers.Tokenizer: id})
def load_gpt2():
    generator = pipeline('text-generation', model='gpt2')

    return generator

@st.cache(hash_funcs={tokenizers.Tokenizer: id})
def load_bert():
    unmasker = pipeline('fill-mask', model='bert-base-cased')
    return unmasker


en2de  = load_en2de()
tokenizer_de2en, de2en = load_de2en()
unmasker = load_bert()
generator = load_gpt2()

en_to_de_output = en2de(user_input)
translated_text = en_to_de_output[0]['translation_text']
# st.write("De text->", translated_text)
input_ids = tokenizer_de2en.encode(translated_text, return_tensors="pt")
output_ids = de2en.generate(input_ids)[0]
augmented_text = tokenizer_de2en.decode(output_ids, skip_special_tokens=True)

st.write("**Translated Sentence->**",augmented_text)

orig_split = user_input.split()
inp_split = user_input.split()
len_input = len(user_input.split())
rand_idx = random.randint(0,len_input-1)
inp_split[rand_idx] = '[MASK]'

rand_idx2 = random.randint(1,len_input-2)

new_list = orig_split[:rand_idx2] + ['[MASK]'] + orig_split[rand_idx2:]
new_mask_sent = ' '.join(new_list)

mask_sent = ' '.join(inp_split)
show_debug = st.sidebar.checkbox("Debug", False) #True
show_all = st.sidebar.checkbox("Show all", False) #True
show_detail = st.sidebar.checkbox("Show details", False)

if show_debug:
    st.write("Masked sentence->",mask_sent)
    st.write("Masked insert sentence->",new_mask_sent)

num_show =2 
if show_all:
    num_show = 5

unmask_sent = unmasker(mask_sent)
unmask_sent2 = unmasker(new_mask_sent)

bert_insert = []
for res in unmask_sent:
    bert_insert.append(res["sequence"])

bert_rep = []
for res in unmask_sent2:
    bert_rep.append(res["sequence"])

st.write("**BERT Replace Idx->**",rand_idx,bert_insert[:num_show])
st.write("**BERT Insert Idx->**",rand_idx2,bert_rep[:num_show])


output_length = len_input + 5
gpt_sent = generator(user_input, max_length=output_length, num_return_sequences=5)

st.write("**GPT2 Output->**",gpt_sent[:num_show])

if show_detail:

    st.write("**BERT Replace->**",unmask_sent[:num_show])
    st.write("**BERT Insert->**",unmask_sent2[:num_show])






