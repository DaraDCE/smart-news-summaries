# Angola News Summariser App

This is a Streamlit app that generates summaries of news articles from various RSS feeds in Angola. It uses the Llama-index library for indexing and querying the news articles, the Langchain library to manage execution chains, and calls a large language model (LLM) to generate summaries and chat responses.

## Dependencies

The app has the following key dependencies:

- `feedparser`
- `streamlit`
- `streamlit_chat`
- `llama_index`
- `nltk`
- `langchain`

## Usage

Once the app is running, you may interact with it via the chat interface. Enter your question or topic in the input field and click the submit button. The app will generate a summary of three news articles related to your query.

The app fetches news articles from various RSS feeds specific to Angola. It sanitizes and parses the feeds, and builds an index from the extracted articles. Llama-index is used for indexing and querying the articles, configured as a Langchain custom tool.

Langchain provides an agent executor which determines whether the associated LLM should respond conversationally to user input, or call the Llama-index news summariser tool. The LLM is the OpenAI GPT-3.5-turbo model.

The app keeps track of the chat history and displays it in the user interface.

## Notes

- The app uses a caching mechanism to improve performance and prevent unnecessary re-running of certain operations. The caching is performed with Streamlit's `st.cache` decorator.
- The app is designed to summarize news articles related to Angola, but it can be customized to work with news articles from different sources or topics by modifying the RSS feeds and index configuration.
- A simple regex and nltk based keyword index is used to minimise LLM token usage. List and vector indicies perform better, however have a substantially higher token cost.
- The app requires an API key for the OpenAI GPT-3.5-turbo model. Make sure to set the `OPENAI_API_KEY` environment variable with your API key before running the app.

## Acknowledgments

This app is built upon various open-source libraries and models, the key ones being:

- Streamlit: A lightweight Python library for building and deploying interactive web apps.
- Streamlit_chat: A Python library that provides a simple chat interface for Streamlit.
- Langchain: A Python library that supports building composable LLM apps.
- Llama-index: A Python library for building and querying indexes over diverse data types and sources.
- OpenAI GPT-3.5-turbo: An OpenAI large language model.

This app was inspired by these Medium.com contributors:

- Timothy Mugayi
- Heiko Hotz

## License

This app is released under the MIT License.
