import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit.components.v1 as components
import urllib.parse
import random

# Sivun asetukset
st.set_page_config(
    page_title="Takakansi-suosittelija", 
    page_icon="📚",
    layout="centered"
)

# Tyylittelyä
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #ddd;
    }
    </style>
    """, unsafe_allow_html=True)

RSS_URL = "https://feeds.soundcloud.com/users/soundcloud:users:381159740/sounds.rss"

# --- DATAN HAKU ---
@st.cache_data(ttl=86400) # Välimuisti tyhjenee kerran vuorokaudessa
def fetch_episodes():
    feed = feedparser.parse(RSS_URL)
    episodes = []
    for entry in feed.entries:
        # Puhdistetaan kuvaus HTML-tägeistä (varmuuden vuoksi)
        summary = entry.summary.replace('<p>', '').replace('</p>', '\n').replace('<br />', '\n')
        
        episodes.append({
            "title": entry.title,
            "summary": summary,
            "link": entry.link,
            "search_text": f"{entry.title}. {summary}"
        })
    return episodes

# --- TEKOÄLYMALLI ---
@st.cache_resource
def load_model():
    # Laadukas monikielinen malli
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

episodes = fetch_episodes()
model = load_model()

@st.cache_resource
def get_embeddings(_episodes):
    texts = [e['search_text'] for e in _episodes]
    return model.encode(texts, convert_to_tensor=True)

episode_embeddings = get_embeddings(episodes)

# --- KÄYTTÖLIITTYMÄ ---
st.title("📚 Takakansi-podcast")
st.subheader("Löydä kuunneltavaa tekoälyn avulla")
st.write("Tämä sovellus auttaa sinua löytämään satojen jaksojen joukosta ne, jotka käsittelevät sinua kiinnostavia aiheita.")

# Sivupalkki asetuksille
with st.sidebar:
    st.header("Asetukset")
    top_k = st.slider("Suositusten määrä", min_value=1, max_value=10, value=3)
    st.divider()
    st.write("Takakansi on kirjallisuusaiheinen podcast, jota isännöi Markus Markus-Lyra.")
    if st.button("🎲 Ehdota satunnaista jaksoa"):
        random_idx = random.randint(0, len(episodes)-1)
        st.session_state.random_ep = episodes[random_idx]

# Hakukenttä
user_input = st.text_input(
    "Syötä aihe, avainsanoja tai lista kiinnostuksen kohteista:", 
    placeholder="esim. filosofia, historia, kirjoittaminen, scifi..."
)

def display_episode(episode, score=None):
    score_text = f" (Sopivuus: {int(score*100)}%)" if score is not None else ""
    with st.expander(f"📖 {episode['title']}{score_text}"):
        st.write(episode['summary'])
        
        # SoundCloud Embed
        encoded_url = urllib.parse.quote(episode['link'])
        embed_code = f"""
        <iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" 
        src="https://w.soundcloud.com/player/?url={encoded_url}&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true">
        </iframe>
        """
        components.html(embed_code, height=180)
        st.write(f"[Avaa suoraan SoundCloudissa]({episode['link']})")

# Näytetään satunnainen jakso jos painiketta painettu
if 'random_ep' in st.session_state and not user_input:
    st.info("Tässäpä mielenkiintoinen jakso:")
    display_episode(st.session_state.random_ep)

# Hakulogiikka
if user_input:
    with st.spinner('Etsitään parhaita jaksoja...'):
        query_embedding = model.encode(user_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, episode_embeddings)[0]
        
        # Haetaan käyttäjän haluama määrä tuloksia
        top_results = torch.topk(cos_scores, k=min(top_k, len(episodes)))
        
        st.success(f"Löysin seuraavat jaksot aiheelle '{user_input}':")
        
        for score, idx in zip(top_results[0], top_results[1]):
            display_episode(episodes[idx], score=score)

else:
    if 'random_ep' not in st.session_state:
        st.info("Kirjoita aihe hakuun tai kokeile onneasi sivupalkin satunnaispainikkeella.")

st.divider()
st.caption(f"Mukana yhteensä {len(episodes)} Takakansi-podcastin jaksoa. Data haettu SoundCloud RSS:stä.")