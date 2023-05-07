import time # assess performance

script_start_time = time.perf_counter()

# core app dependencies
import feedparser
import streamlit as st
from datetime import datetime, date, timedelta, timezone
from fuzzywuzzy import fuzz
from unidecode import unidecode
import re
import urllib.parse
#from PIL import Image
import concurrent.futures
import os # for calling environmental variables locally

# for st chatbot frame
from streamlit_chat import message

# llama index
from llama_index import (
    GPTSimpleVectorIndex,
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
    PromptHelper,
    MockLLMPredictor
)
from llama_index.prompts.prompts import SummaryPrompt, QuestionAnswerPrompt
from llama_index.readers import Document # to create Document objects
from llama_index.node_parser import SimpleNodeParser # to parse Document objects to nodes

# langchain prompts and chain management
from langchain import (
    PromptTemplate, # for customsing prompt templates
    LLMChain # for simple prompt / LLM chain
)
#from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain

# import LLMs
from langchain.chat_models import ChatOpenAI
from langchain import HuggingFaceHub


## Fetch API keys (local)
# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Huggingface Hub
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HFHUB_API_TOKEN_DOLLY_RSS')
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

start_time = time.perf_counter()

## Define function to build feed lists
@st.cache_data # cache to prevent re-running
def build_feeds():

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
    
    return rss_feeds, allowed_domains

rss_feeds, allowed_domains = build_feeds()

duration = time.perf_counter() - start_time
print(f"Initiate feeds: {duration}")

start_time = time.perf_counter()

# Title
st.title(f'Notícias Angolanas :flag-ao: :newspaper: :bulb:')

# Create two columns with widgets for keyword and news source filters
col1, col2 = st.columns(
    [2, 2],
    gap="medium"
    )
with col1:
    keyword = st.text_input("Procurar palavra-chave", value='')
with col2:
    news_source = st.selectbox("Selecione uma fonte", [''] + list(sorted(rss_feeds.keys())))

duration = time.perf_counter() - start_time
print(f"Set title and static widgets: {duration}")

start_time = time.perf_counter()

## Define a function to validate, sanitize and parse an RSS feed
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

duration = time.perf_counter() - start_time
print(f"Validate, sanitize and parse feeds: {duration}")

start_time = time.perf_counter()

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

duration = time.perf_counter() - start_time
print(f"Fetch and store feeds: {duration}")

start_time = time.perf_counter()

### Prepare data for indexing
## Convert dictionary-like object into list of dictionaries with llama indexing key labelling convention
# pass returned news titles and summaries to list of dictionaries
news_list_of_dicts = []
for i, (source, item) in enumerate(filtered_articles[:-1]):
    news_list_of_dicts.append({'doc_id' : item['title'], 'text' : item['summary']})

duration = time.perf_counter() - start_time
print(f"Prepare data for indexing {duration}")


start_time = time.perf_counter()

#### Define LLM for building and querying index:
### General settings
## define prompt helper
# set maximum input size
max_input_size = 4096
# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

## define custom QuestionAnswerPrompt
query_str = "Were there news about the MPLA today?"
QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please search related articles and answer in Portuguese: {query_str} \n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

# define mock LLM predictor and service context to pre-count tokens
mock_llm_predictor = MockLLMPredictor(max_tokens=256)
mock_service_context = ServiceContext.from_defaults(llm_predictor=mock_llm_predictor)

### databricks/dolly-v2-3b
## See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for more models
#repo_id = "databricks/dolly-v2-3b" 
#llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
#llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0})

# incorporate llm for llama index predictions
#llm_predictor = LLMPredictor(llm=llm) # dolly
#service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                               #prompt_helper=prompt_helper, # try defaults first
                                               #chunk_size_limit=512 # try defaults first
                                               #)
## openai/gpt-3.5-turbo
# define LLM to use (gpt-3.5-turbo == $0.002 / 1K tokens)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# incorporate llm for llama index predictions
llm_predictor = LLMPredictor(llm=llm)
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor,
                                               prompt_helper=prompt_helper,
                                               chunk_size_limit=512)

duration = time.perf_counter() - start_time
print(f"Set LLM: {duration}")

start_time = time.perf_counter()

#### Define function to set index type and build
### Simple vector index (best for semantic / top-k search; ie. queries 'cherry-pick' indexed data)
## Create a llama-index object from list of documents in Document format
#@st.cache_resource # cache function to prevent rerun on user input
def build_index(news_list_of_dicts, service_context, use_async=True):
    
    # convert list of dictionaries to list of documents with Document format for indexing
    documents = []
    for article in news_list_of_dicts:
        documents.append(Document(doc_id = article['doc_id'], text = article['text']))
        
    # parse Documents objects to nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # build index
    vector_index = GPTSimpleVectorIndex.from_documents(nodes, # "documents" or "nodes"
                                                       service_context=service_context,
                                                       use_async=use_async # improve performance
                                                       )
    return vector_index

vector_index = build_index(news_list_of_dicts, mock_service_context) # build index

duration = time.perf_counter() - start_time
print(f"Build index: {duration}")

start_time = time.perf_counter()

### Save and reload index once for reuse
## To streamlit session state
# Save the index to streamlit session state
if 'index' not in st.session_state:
    st.session_state.index = vector_index

# Load the index from streamlit session state
index = st.session_state.index

duration = time.perf_counter() - start_time
print(f"Save and reload index: {duration}")


start_time = time.perf_counter()

## Chat settings
# Define function to pass user input to llm and query index
def conversational_chat(user_question, QA_PROMPT, service_context):

    llm_response = index.query(
        user_question,
        text_qa_template=QA_PROMPT,
        service_context=service_context,
        similarity_top_k=5
        ) 

    extracted_response = f"{llm_response}"

    #st.session_state['history'].append((user_question, llm_response["answer"]))
    #return llm_response["answer"]
    st.session_state['history'].append((user_question, extracted_response))
    return extracted_response

## Define initial chat session state and containers
# Session states
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Olá! Posso responder perguntas sobre as notícias recentes. Pergunte!" + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["..."]
        
# Containers for the chat history
response_container = st.container() # chat history
container = st.container() # user inputs

## Load Chat UI on session start
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input(label="Faça uma pergunta sobre as notícias:", placeholder="...", key='input')
        submit_button = st.form_submit_button(label='Envia')

    if submit_button and user_input:
        output = conversational_chat(user_input, QA_PROMPT, mock_service_context) # not ideal, revisit

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# display session chat sequence
if st.session_state['generated']:
    with response_container:
        #for i in range(len(st.session_state['generated'])):
            #message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            #message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        message(st.session_state["past"][-1], is_user=True, key='user', avatar_style="open-peeps", seed="Harley")
        message(st.session_state["generated"][-1], key='generated', avatar_style="bottts", seed="Mia")

duration = time.perf_counter() - start_time
print(f"Initiate and load chatbot: {duration}")


start_time = time.perf_counter()

## Format articles dates for sorting and display
# Define a function to convert the date string into a datetime object and handle exceptions in rss item labels
def get_date_published(item):
    try:
        return datetime(*item.published_parsed[:6])
    except AttributeError:
        return datetime(*item.updated_parsed[:6]) # specific to "rss.dw.com"

# Sort the filtered news articles by date (with equal weighting of source in key)
filtered_articles = sorted(filtered_articles, key=lambda x: (get_date_published(x[1]), x[0]), reverse=True)


### Display filtered news articles with buttons to increment upto max 20 and reset
## Display after rss articles parsed
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

duration = time.perf_counter() - start_time
print(f"Format dates and display articles {duration}")

total_duration = time.perf_counter() - script_start_time
print(f"Total runtime: {total_duration}")
print("\n")