import feedparser
import streamlit as st
from datetime import datetime, date, timedelta, timezone
from fuzzywuzzy import fuzz
from unidecode import unidecode
import re
import urllib.parse
from PIL import Image
import concurrent.futures

# List of RSS feeds
rss_feeds = {
    #"Club-K": "https://www.club-k.net/index.php?option=com_obrss&task=feed&id=2:rss-noticias-do-club-k&format=feed&lang=pt", # bad url
    #"O País": "https://opais.co.ao/feed/", # sign-in required
    "Africa Intelligence": "http://feeds.feedburner.com/AfricaIntelligence",
    "Correio da Kianda": "https://correiokianda.info/feed/",
    "Notícias de Angola": "https://noticiasdeangola.co.ao/feed/",
    "Folha 8": "https://jornalf8.net/feed/",
    "Imparcial Press": "https://imparcialpress.net/feed/",
    "Correio Angolense": "https://www.correioangolense.co.ao/feed/",
    "Maka Angola": "https://www.makaangola.org/feed/",
    "Camunda News": "https://camundanews.com/feed",
    #"Valor Económico": "https://valoreconomico.co.ao/rss.php", # bad feed
    "Angola Online": "https://angola-online.net/noticias.xml",
    "DW Africa": "https://rss.dw.com/rdf/rss-br-africa"
}

# List of allowed domains for RSS feeds
allowed_domains = [
#"club-k.net", # bad url
#"opais.co.ao", # sign-in required
"feeds.feedburner.com",
"correiokianda.info",
"noticiasdeangola.co.ao",
"jornalf8.net",
"imparcialpress.net",
"www.correioangolense.co.ao",
"www.makaangola.org",
"camundanews.com",
#"valoreconomico.co.ao", # bad feed
"angola-online.net",
"rss.dw.com"
]

# Title
st.title(f'Notícias Angolanas :flag-ao: :newspaper: :bulb:')

# Widgets for searching by keyword and news source, displayed in column with app (as opposed to 'sidebar')
keyword = st.text_input("Procurar palavra-chave", value='')
news_source = st.selectbox("Selecione uma fonte", [''] + list(sorted(rss_feeds.keys())))

# Define a function to validate, parse, sanitize, and fetch an RSS feed
def parse_feed(source, url):
    try:
        # Validate the URL format
        if not re.match(r"^https?://", url):
            raise ValueError("URL não está em um formato válido")
        
        # Parse the URL to check the domain
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc not in allowed_domains:
            raise ValueError("O domínio não é permitido")

        # Sanitize the URL parameters
        sanitized_url = urllib.parse.urlunparse(parsed_url._replace(query=""))
        sanitized_url = urllib.parse.quote(sanitized_url, safe=':/')
        
        # Use HTTPS
        if parsed_url.scheme == "http":
            sanitized_url = "https://" + parsed_url.netloc + parsed_url.path
        
        feed = feedparser.parse(sanitized_url)

        return source, feed

    except Exception as e:
        print(f"Houve um erro ao interpretar a fonte {source}: {e}")
        return None

# Create a thread pool for concurrent processing and extraction of rss feeds, including source and keyword filtering
with concurrent.futures.ThreadPoolExecutor(max_workers=len(rss_feeds)) as executor:
    # Submit the parsing of each feed to the pool
    futures = {executor.submit(parse_feed, source, url): (source, url) for source, url in rss_feeds.items()}
    
    filtered_articles = []

    for future in concurrent.futures.as_completed(futures):
        source, feed = futures[future]
        result = future.result()
        if result is not None:
            source, feed = result
            if news_source and source != news_source:
                continue
            if "bozo_exception" in feed:
                #st.warning(f"Could not fetch news from {source} ({url}), please try again later.")
                print(f"Não foi possível obter notícias de {source} ({url}), tente mais tarde.")
                continue
            for item in feed["entries"]:
                # Filter by keyword with fuzzy search
                if keyword:
                    title_score = fuzz.token_set_ratio(unidecode(keyword.lower()), unidecode(item.title.lower()))
                    summary_score = fuzz.token_set_ratio(unidecode(keyword.lower()), unidecode(item.summary.lower()))
                    if title_score < 80 and summary_score < 80:
                        continue
                filtered_articles.append((source, item))

# Define a function to convert the date string into a datetime object and handle exceptions in rss item labels
def get_date_published(item):
    try:
        return datetime(*item.published_parsed[:6])
    except AttributeError:
        return datetime(*item.updated_parsed[:6]) # specific to "rss.dw.com"

# Sort the filtered news articles by date (with equal weighting of source in key)
filtered_articles = sorted(filtered_articles, key=lambda x: (get_date_published(x[1]), x[0]), reverse=True)

# Display the filtered news articles with buttons to increment upto max 20 and reset
if filtered_articles:
    articles_displayed = 10
    for i, (source, item) in enumerate(filtered_articles[:articles_displayed]):
        st.write(f"## {item['title']}")
        st.write(f"**Fonte:** {source},  " + f"**Publicado:** {item['published'] if 'published' in item else item['updated']}")
        #st.markdown(item["summary"]) # exclude markdown summary
        st.write(f"Leia mais: {item['link']}")
    
    if len(filtered_articles) > 10:
        load_more = st.button('Carrega mais')
        if load_more:
            articles_displayed += 10
            if articles_displayed > 20:
                articles_displayed = 20
                st.write("_Não há mais notícias para carregar._")
                
            else:
                for i, (source, item) in enumerate(filtered_articles[10:articles_displayed], start=10):
                    st.write(f"## {item['title']}")
                    st.write(f"**Fonte:** {source},  " + f"**Publicado:** {item['published'] if 'published' in item else item['updated']}")
                    #st.markdown(item["summary"]) # exclude markdown summary
                    st.write(f"Leia mais: {item['link']}")

            st.write(f"{articles_displayed} notícias exibidas.")
            if articles_displayed == 20:
                st.write("_Não há mais notícias para carregar._")
        else:
            st.write(f"{articles_displayed} notícias exibidas.")
    else:
        st.write("_Não foram encontradas mais notícias relacionadas às palavras-chave ou a fonte._")

    if articles_displayed > 10:
        reset = st.button('Voltar')
        if reset:
            articles_displayed = 10
            st.experimental_rerun()
else:
    st.write("_Não foram encontradas mais notícias relacionadas às palavras-chave ou a fonte._")