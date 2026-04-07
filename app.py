import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
from docx import Document
import requests
from bs4 import BeautifulSoup
import re
import time  

# 1. CẤU HÌNH & LOAD MODEL
st.set_page_config(page_title="LDA + TF-IDF", layout="wide")

@st.cache_resource
def load_lda_resources():
    path_model = "/Users/trandaiphat/data_spark/LDA_Model_7M_Final.pkl"
    with open(path_model, "rb") as f:
        model_data = pickle.load(f)
    
    acc_df = model_data['accuracy_report']
    sorted_acc = acc_df.sort_values(by='count', ascending=False)
    
    mapping = {}
    used_cats = set()
    used_topics = set()
    
    for _, row in sorted_acc.iterrows():
        t_id, cat = row['pred_topic'], row['category']
        if t_id not in used_topics and cat not in used_cats:
            mapping[t_id] = cat
            used_topics.add(t_id)
            used_cats.add(cat)
            
    return model_data, mapping

data, topic_mapping = load_lda_resources()
vocab = data['vocab']
topics_matrix = data['topics_matrix']

# 2. LOGIC PHÂN TÍCH & STREAMING
def stream_data(text):
    """Hàm tạo luồng chạy chữ cho Chatbot"""
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.04)

def predict_topic_logic(text):
    text_clean = text.lower()
    all_tokens = re.findall(r'\w+', text_clean)
    if not all_tokens: return None, None, None

    words_in_doc = []
    i = 0
    while i < len(all_tokens):
        if i < len(all_tokens) - 1:
            bigram = f"{all_tokens[i]}_{all_tokens[i+1]}" 
            bigram_space = f"{all_tokens[i]} {all_tokens[i+1]}"
            
            if bigram in vocab:
                words_in_doc.append(bigram)
                i += 2
                continue
            elif bigram_space in vocab:
                words_in_doc.append(bigram_space)
                i += 2
                continue
        
        if all_tokens[i] in vocab:
            words_in_doc.append(all_tokens[i])
        i += 1

    if not words_in_doc: return None, None, None

    topic_probs = [0.0] * 4
    word_counts = pd.Series(words_in_doc).value_counts()
    found_words_data = []
    
    for word, count in word_counts.items():
        idx = vocab.index(word)
        weights = topics_matrix[idx]
        topic_probs = [p + (weights[k] * count) for k, p in enumerate(topic_probs)]
        
        best_t_idx = weights.argmax()
        found_words_data.append({
            "Từ khóa": word, 
            "Số lần": count, 
            "Chủ đề": topic_mapping.get(best_t_idx, "Khác"),
            "Loại": "Từ ghép" if (" " in word or "_" in word) else "Từ đơn"
        })

    total_prob = sum(topic_probs)
    if total_prob > 0:
        topic_probs = [p/total_prob for p in topic_probs]
    
    return topic_probs, words_in_doc, pd.DataFrame(found_words_data)

# 3. HÀM XỬ LÝ PHẢN HỒI CHATBOT
def get_ai_response(prompt, context_text):
    probs, words, word_df = predict_topic_logic(context_text)
    if probs is None: 
        return "Văn bản này không chứa từ nào trong bộ từ điển 7 triệu dòng của tớ cả!", None, None
    
    p_lower = prompt.lower()
    max_idx = probs.index(max(probs))
    predicted_topic = topic_mapping.get(max_idx, "Không xác định")

    if any(x in p_lower for x in ["%", "tỉ lệ", "tỷ lệ", "xác suất", "phần trăm"]):
        df_probs = pd.DataFrame({
            "Chủ đề": [topic_mapping.get(i) for i in range(4)],
            "Độ tin cậy": [f"{p*100:.2f}%" for p in probs]
        }).sort_values("Độ tin cậy", ascending=False)
        
        ans = f"Đây là bảng phân tích xác suất. Chủ đề **{predicted_topic}** có độ chính xác cao nhất!"
        return ans, None, df_probs

    if any(x in p_lower for x in ["từ nào", "từ khóa", "tại sao", "cơ sở"]):
        target_topic = predicted_topic
        for t_name in topic_mapping.values():
            if t_name.lower() in p_lower: target_topic = t_name
        
        df_target = word_df[word_df['Chủ đề'] == target_topic].copy()
        if df_target.empty:
            return f"Tớ không thấy từ khóa đặc trưng nào của nhóm **{target_topic}**.", None, None

        df_target["Từ khóa"] = df_target["Từ khóa"].str.replace("_", " ")
        top_single = df_target[df_target['Loại'] == "Từ đơn"].sort_values("Số lần", ascending=False).head(5)
        top_double = df_target[df_target['Loại'] == "Từ ghép"].sort_values("Số lần", ascending=False).head(5)
        final_evidence = pd.concat([top_double, top_single])
        
        ans = f"Đây là bằng chứng từ vựng thuộc nhóm **{target_topic}** tớ tìm thấy. Các từ này có trọng số rất cao đấy!"
        return ans, None, final_evidence[["Từ khóa", "Số lần", "Loại"]]

    
    if any(x in p_lower for x in ["so sánh", "biểu đồ", "thay vì", "khác gì", "đồ thị"]):
        ans = f"Để Phat dễ hình dung, tớ đã vẽ biểu đồ so sánh trọng số giữa 4 thể loại. Nhìn vào đây cậu sẽ thấy tại sao **{predicted_topic}** lại vượt trội hơn hẳn!"
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [topic_mapping.get(i) for i in range(4)]
        sns.barplot(x=labels, y=probs, ax=ax, palette="coolwarm")
        ax.set_ylabel("Mức độ tương quan")
        return ans, fig, None

    if any(x in p_lower for x in ["là gì", "thể loại", "chủ đề", "phân loại", "đoán"]):
        ans = f"Kết quả phân tích: Bài báo này thuộc thể loại **{predicted_topic}**. Tớ đã vẽ biểu đồ xác suất chi tiết bên dưới cho cậu!"
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = [topic_mapping.get(k) for k in range(4)]
        sns.barplot(x=labels, y=probs, ax=ax, palette="viridis")
        ax.set_ylabel("Xác suất")
        return ans, fig, None 

    return f"Tớ đã sẵn sàng! Phat muốn tớ **in bảng % xác suất**, **liệt kê từ khóa chính**, hay **vẽ biểu đồ so sánh** các thể loại?", None, None

# 4. GIAO DIỆN STREAMLIT
if "messages" not in st.session_state: st.session_state.messages = []
if "context_text" not in st.session_state: st.session_state.context_text = ""

with st.sidebar:
    st.header("Nguồn Dữ Liệu")
    mode = st.radio("Chọn nguồn nhập:", ["Văn bản", "File (PDF/Docx)", "URL Bài báo"])
    raw = ""
    
    if mode == "Văn bản":
        raw = st.text_area("Dán nội dung vào đây:", height=300)
    elif mode == "File (PDF/Docx)":
        f = st.file_uploader("Tải file lên", type=["pdf", "docx"])
        if f:
            if f.type == "application/pdf":
                raw = " ".join([p.extract_text() for p in PdfReader(f).pages])
            else:
                raw = " ".join([para.text for para in Document(f).paragraphs])
    elif mode == "URL Bài báo":
        u = st.text_input("Dán link bài báo:")
        if u:
            try:
                res = requests.get(u)
                raw = " ".join([p.get_text() for p in BeautifulSoup(res.text, 'html.parser').find_all('p')])
            except:
                st.error("Không thể lấy dữ liệu từ URL này!")

    if st.button("Xử lý dữ liệu"):
        if raw.strip():
            st.session_state.context_text = raw
            st.session_state.messages = [] 
            st.success("Nạp dữ liệu thành công")
        else:
            st.warning("Cậu chưa nhập gì cả!")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "chart" in m: st.pyplot(m["chart"])
        if "table" in m: st.table(m["table"])

if prompt := st.chat_input("Hỏi tớ về bài báo này..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): 
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.context_text:
            st.error("Phat ơi, nạp bài báo ở bên trái trước đã nhé!")
        else:
            ans, chart, table = get_ai_response(prompt, st.session_state.context_text)
            
            full_response = st.write_stream(stream_data(ans))

            if chart is not None:
                st.pyplot(chart)
            if table is not None:
                st.table(table)
            
            msg_entry = {"role": "assistant", "content": full_response}
            if chart is not None: msg_entry["chart"] = chart
            if table is not None: msg_entry["table"] = table
            st.session_state.messages.append(msg_entry)