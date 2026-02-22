import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import torch

# Säädetään sivun ulkoasua
st.set_page_config(page_title="Takakansi-suosittelija", page_icon="📚")

st.title("📚 Takakansi-podcast -suosittelija")
st.write("Syötä aihe tai kirjailija, niin etsin sinulle parhaiten sopivat jaksot.")

RSS_URL = "https://feeds.soundcloud.com/users/soundcloud:users:381159740/sounds.rss"

# 1. Haetaan jaksot RSS-syötteestä (välimuistissa, jotta on nopea)
@st.cache_data
def fetch_episodes():
    feed = feedparser.parse(RSS_URL)
    episodes = []
    for entry in feed.entries:
        # Yhdistetään otsikko ja kuvaus hakua varten
        content_to_index = f"{entry.title}. {entry.summary}"
        episodes.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link,
            "search_text": content_to_index
        })
    return episodes

# 2. Ladataan tekoälymalli (käytetään monikielistä mallia, joka osaa suomea)
@st.cache_resource
def load_model():
    # Tämä malli on kevyt ja ymmärtää suomea erinomaisesti
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

episodes = fetch_episodes()
model = load_model()

# 3. Luodaan vektorit (embeddings) jaksoista (tämä tehdään vain kerran)
@st.cache_resource
def get_embeddings(_episodes):
    texts = [e['search_text'] for e in _episodes]
    return model.encode(texts, convert_to_tensor=True)

episode_embeddings = get_embeddings(episodes)

# 4. Käyttöliittymä hakuun
user_input = st.text_input("Mistä aiheesta haluaisit kuunnella?", placeholder="esim. scifi, historia, luovuus tai Mika Waltari")

if user_input:
    # Muutetaan käyttäjän haku vektoriksi
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    
    # Lasketaan samankaltaisuus kaikkien jaksojen välillä
    cos_scores = util.cos_sim(query_embedding, episode_embeddings)[0]
    
    # Haetaan 5 parasta tulosta
    top_results = torch.topk(cos_scores, k=min(5, len(episodes)))
    
    st.subheader(f"Suositukset aiheelle: '{user_input}'")
    
    for score, idx in zip(top_results[0], top_results[1]):
        episode = episodes[idx]
        with st.expander(f"{episode['title']} (Match: {score:.2f})"):
            st.write(episode['summary'])
            st.video(episode['link']) # Streamlit osaa näyttää SoundCloud-linkit soittimena
            st.write(f"[Lue lisää ja kuuntele täällä]({episode['link']})")

else:
    st.info("Kirjoita jotain yllä olevaan kenttään aloittaaksesi.")
    st.write(f"Sovelluksessa on mukana {len(episodes)} jaksoa.")