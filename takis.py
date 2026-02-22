import streamlit as st
import feedparser
from sentence_transformers import SentenceTransformer, util
import torch
import streamlit.components.v1 as components # Lisätty HTML-komponentteja varten
import urllib.parse

st.set_page_config(page_title="Takakansi-suosittelija", page_icon="📚")

st.title("📚 Takakansi-podcast -suosittelija")

RSS_URL = "https://feeds.soundcloud.com/users/soundcloud:users:381159740/sounds.rss"

@st.cache_data
def fetch_episodes():
    feed = feedparser.parse(RSS_URL)
    episodes = []
    for entry in feed.entries:
        # Haetaan suora mp3-linkki enclosure-kentästä
        audio_url = ""
        if 'enclosures' in entry and len(entry.enclosures) > 0:
            audio_url = entry.enclosures[0].href
            
        content_to_index = f"{entry.title}. {entry.summary}"
        episodes.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link,
            "audio_url": audio_url,
            "search_text": content_to_index
        })
    return episodes

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

episodes = fetch_episodes()
model = load_model()

@st.cache_resource
def get_embeddings(_episodes):
    texts = [e['search_text'] for e in _episodes]
    return model.encode(texts, convert_to_tensor=True)

episode_embeddings = get_embeddings(episodes)

user_input = st.text_input("Mistä aiheesta haluaisit kuunnella?", placeholder="esim. scifi, historia, luovuus tai Mika Waltari")

if user_input:
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, episode_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(5, len(episodes)))
    
    st.subheader(f"Suositukset aiheelle: '{user_input}'")
    
    for score, idx in zip(top_results[0], top_results[1]):
        episode = episodes[idx]
        with st.expander(f"{episode['title']} (Match: {score:.2f})"):
            st.write(episode['summary'])
            
            # --- TAPA A: Streamlitin oma audiosoitin (toimii varmasti) ---
            if episode['audio_url']:
                st.audio(episode['audio_url'])
            
            # --- TAPA B: SoundCloud Embed (Näyttää hienolta) ---
            # Luodaan iframe-koodi SoundCloudin embed-linkistä
            encoded_url = urllib.parse.quote(episode['link'])
            embed_code = f"""
            <iframe width="100%" height="166" scrolling="no" frameborder="no" allow="autoplay" 
            src="https://w.soundcloud.com/player/?url={encoded_url}&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true">
            </iframe>
            """
            components.html(embed_code, height=180)
            
            st.write(f"[Avaa SoundCloudissa]({episode['link']})")
else:
    st.info("Kirjoita jotain yllä olevaan kenttään aloittaaksesi.")