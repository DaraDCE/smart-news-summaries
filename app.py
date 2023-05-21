import time # assess performance

script_start_time = time.perf_counter()

# core app dependencies
import feedparser
import streamlit as st
from datetime import datetime, date, timedelta, timezone
from unidecode import unidecode
import re
import urllib.parse
import concurrent.futures
import os
import socket # to manage timeouts

# for st chatbot frame
from streamlit_chat import message

# llama index
from llama_index import (
    GPTSimpleKeywordTableIndex,
    LLMPredictor,
    ServiceContext,
    PromptHelper,
    MockLLMPredictor,
    LangchainEmbedding
)
from llama_index.prompts.prompts import QuestionAnswerPrompt
from llama_index.readers import Document # to create Document objects
from llama_index.node_parser import SimpleNodeParser # to parse Document objects to nodes

# langchain tool abstractions
from llama_index.langchain_helpers.agents import (
    IndexToolConfig,
    LlamaIndexTool
)

# langchain prompts, chain, tool, and agent management
from langchain import (
    PromptTemplate, # for customising prompt templates
    LLMChain # for simple prompt / LLM chain
)

from langchain.agents import (
    AgentType,
    initialize_agent,
    Tool
)

from langchain.memory import ConversationBufferMemory

# import LLMs
from langchain.chat_models import ChatOpenAI

# nltk for keywords
import nltk

## Fetch API keys (local)
# OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


start_time = time.perf_counter()

## Define function to build feed lists
@st.cache_data # cache to prevent re-running
def build_feeds():

    # List of RSS feeds
    rss_feeds = {
        "Club-K": "https://www.club-k.net/index.php?option=com_obrss&task=feed&id=2:rss-noticias-do-club-k&format=feed&lang=pt", # bad url
        #"O País": "https://opais.co.ao/feed/", # sign-in required
        "Africa Intelligence": "http://feeds.feedburner.com/AfricaIntelligence",
        "Correio da Kianda": "https://correiokianda.info/feed/",
        "Notícias de Angola": "https://noticiasdeangola.co.ao/feed/",
        "Folha 8": "https://jornalf8.net/feed/",
        "Imparcial Press": "https://imparcialpress.net/feed/",
        "Correio Angolense": "https://www.correioangolense.co.ao/feed/",
        "Maka Angola": "https://www.makaangola.org/feed/",
        "Camunda News": "https://camundanews.com/feed",
        "Valor Económico": "https://valoreconomico.co.ao/rss.php", # bad feed
        "Angola Online": "https://angola-online.net/noticias.xml",
        "DW Africa": "https://rss.dw.com/rdf/rss-br-africa"
    }

    # List of allowed domains for RSS feeds
    allowed_domains = [
    "www.club-k.net", # bad url
    #"opais.co.ao", # sign-in required
    "feeds.feedburner.com",
    "correiokianda.info",
    "noticiasdeangola.co.ao",
    "jornalf8.net",
    "imparcialpress.net",
    "www.correioangolense.co.ao",
    "www.makaangola.org",
    "camundanews.com",
    "valoreconomico.co.ao", # bad feed
    "angola-online.net",
    "rss.dw.com"
    ]
    
    return rss_feeds, allowed_domains

rss_feeds, allowed_domains = build_feeds()

duration = time.perf_counter() - start_time
print(f"Initiate feeds: {duration}")

start_time = time.perf_counter()

# Title
st.title(f'Resumos das notícias de Angola! :flag-ao: :newspaper: :bulb:')

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
            raise ValueError(f"{source}: O domínio não é permitido")

        # Sanitize the URL parameters
        sanitized_url = urllib.parse.urlunparse(parsed_url._replace(query=""))
        sanitized_url = urllib.parse.quote(sanitized_url, safe=':/')
        
        # Use HTTPS
        if parsed_url.scheme == "http":
            sanitized_url = "https://" + parsed_url.netloc + parsed_url.path
        
        # Set a timeout on the socket
        socket.setdefaulttimeout(5)  # Set a timeout of 5 seconds
        
        feed = feedparser.parse(sanitized_url)
        
        if "bozo_exception" in feed:
            raise Exception(f"Erro ao obter feed de {url}: {feed.bozo_exception}")
        
        return source, feed

    except socket.timeout as e:
        print(f"{url}: Tempo limite excedido")
    except Exception as e:
        print(f"{e}")
    return None, None

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
        if result[0] is not None:
            source, feed = result
            for item in feed["entries"]:
                filtered_articles.append((source, item))
        else:
            print(result[1])

duration = time.perf_counter() - start_time
print(f"Fetch and store feeds: {duration}")

start_time = time.perf_counter()

### Prepare data for indexing
@st.cache_data # cache to prevent re-running
def prepare_data(filtered_articles):
    
    # concatenate title, summary and link value fields, paired with new label
    for item in filtered_articles:
        item[1]['title_summary_link'] = item[1]['title'] + item[1]['summary'] + item[1]['link']
        
    # pass returned news titles, summaries and links to list of dictionaries
    news_list_of_dicts = []
    for i, (source, item) in enumerate(filtered_articles):
        news_list_of_dicts.append({'doc_id' : item['title'], 'text' : item['title_summary_link']})
        
    # convert list of dictionaries to list of documents with Document format for indexing
    documents = []
    for article in news_list_of_dicts:
        documents.append(Document(doc_id = article['doc_id'], text = article['text']))
        
    # parse Documents objects to nodes
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    
    return nodes

nodes = prepare_data(filtered_articles)

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

# define custom QuestionAnswerPrompt
QA_SUM_PROMPT_TMPL = (
    "Given the news articles in the provided context, reply to {query_str}:"
    "Reply in the language the question was asked in."
    "Write a summary of the returned news articles."
    "Use a friendly tone, and mention that you have summarised only three articles."
    "Try to use a bullet point per news article, and only use the information provided."
    "Try to include the associated link with the summary in each bullet point.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)
QA_SUM_PROMPT_TMPL = QuestionAnswerPrompt(QA_SUM_PROMPT_TMPL)

# define mock LLM predictor and service context to pre-count tokens
mock_llm_predictor = MockLLMPredictor(max_tokens=256)
mock_service_context = ServiceContext.from_defaults(llm_predictor=mock_llm_predictor)

## openai/gpt-3.5-turbo
# define LLM to use (gpt-3.5-turbo == $0.002 / 1K tokens)
#llm = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")
gpt = ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo")

# configure llm for llama-index
llm=gpt
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
def build_index(nodes, service_context,
    #use_async=True
    ):

    # build index
    simple_keyword_index = GPTSimpleKeywordTableIndex.from_documents(nodes, # "documents" or "nodes"
                                            service_context=service_context,
                                            #use_async=True # improve performance
                                           )
    return simple_keyword_index

simple_keyword_index = build_index(nodes, service_context) # build index

duration = time.perf_counter() - start_time
print(f"Build index: {duration}")

start_time = time.perf_counter()

### Save and reload index once for reuse
## To streamlit session state
# Save the index to streamlit session state
if 'index' not in st.session_state:
    st.session_state.index = simple_keyword_index

# Load the index from streamlit session state
index = st.session_state.index

duration = time.perf_counter() - start_time
print(f"Save and reload index: {duration}")


start_time = time.perf_counter()

### Configure langchain agent with llama-index query engine tool
## Configure llama-index query engine
query_engine = simple_keyword_index.as_query_engine(
    text_qa_template=QA_SUM_PROMPT_TMPL,
    num_chunks_per_query=3, # limit chunks to query, to economise response time and token usage
    service_context=service_context,
    response_mode="tree_summarize"
)

## Configure llama-index query engine as custom langchain tool
tool_config = IndexToolConfig(
    query_engine=query_engine, 
    name=f"News summariser",
    description=f"Useful when you need to select and summarise news articles from keywords.\
    The input to this tool should be the question string, passed exactly as it was received.\
    For example, 'Houve notícias sobre o presidente?'. The output should be in the same language as the input.\
    After the third user question do not run the tool but invite user to come back another day.", # didn't work...
    tool_kwargs={"return_direct": True}
)

tool = LlamaIndexTool.from_tool_config(tool_config)

## Configure langchain agent handler
memory = ConversationBufferMemory(memory_key="chat_history")

conversational_handler = initialize_agent([tool], llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                                          verbose=True, memory=memory)

duration = time.perf_counter() - start_time
print(f"Configure langchain: {duration}")


start_time = time.perf_counter()

## Chat settings
# Define function to pass user input to llm and query index
def chat_handler(user_question):

    try:
        llm_response = conversational_handler.run(user_question)
    
    except Exception as e:

        # prep langchain custom prompt and chain
        template = """Context: {context}
        Statement: Mention briefly in Portuguese that you had difficulty working out an
        answer to the question, explain that you are still learning, and suggest that the
        user tries asking about a subject they're interested in."""

        prompt = PromptTemplate(template=template, input_variables=["context"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        # load exception context
        context = f"While processing the user question {user_question}, exception {e} occurred."
    
        # run exception query
        llm_response = llm_chain.run(context)
        print(f"{e}")

    extracted_response = f"{llm_response}"

    # get number of summarisation tokens used
    print(f"Estimated summarisation token usage: {llm_predictor.last_token_usage}")

    st.session_state['history'].append((user_question, extracted_response))
    return extracted_response

## Define initial chat session state and containers
# Session states
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Olá! Posso resumir as notícias recentes. Posso demorar um minutinho para responder, então tenha paciência!" + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["..."]
        
# Containers for the chat history
response_container = st.container() # chat history
container = st.container() # user inputs

## Load Chat UI on session start
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input(label="Diga sobre o que gostaria de ver notícias:",
            placeholder="Ex. Quais foram as notícias de Luanda?", key='input')
        submit_button = st.form_submit_button(label='Pressione para perguntar')

    if submit_button and user_input:
        output = chat_handler(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

# display session chat sequence
if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="pixel-art-neutral",
                seed="Mia")
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Mia")

duration = time.perf_counter() - start_time
print(f"Initiate and load chatbot: {duration}")


total_duration = time.perf_counter() - script_start_time
print(f"Total runtime: {total_duration}")
print("\n")