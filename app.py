import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("Kikuyu–English Dairy Farming Translator")
st.write("Fine-tuned Meta NLLB-200-distilled-600M model for agricultural extension.")

@st.cache_resource
def load_model():
    model_id = "es-omani/nllb-kikuyu-dairy"
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="kik_Latn", tgt_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, model = load_model()

kikuyu_text = st.text_area(
    "Kikuyu Agricultural Text",
    placeholder="e.g. Ng'ombe ya Friesian nĩ ĩrutaga iria rĩingĩ mũno."
)

if st.button("Translate") and kikuyu_text.strip():
    with st.spinner("Translating..."):
        inputs = tokenizer(kikuyu_text, return_tensors="pt", truncation=True, max_length=128)
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=128
        )
        translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(translation)
