import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm
import praw
import tweepy
from newsapi import NewsApiClient
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import spacy
from transformers import pipeline

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InsuranceDataCollector:
    """Collects and processes insurance data from various sources."""
    
    def __init__(self, config_path: str = "config/data_collection.yaml"):
        """Initialize the data collector."""
        self.config = self._load_config(config_path)
        self.base_dir = Path("data/insurance")
        self._setup_directories()
        self._initialize_apis()
        # Load NLP config from data_collection.yaml or fallback to rag.yaml
        nlp_config = self.config.get('nlp', {})
        if not nlp_config:
            try:
                with open('config/rag.yaml', 'r') as f:
                    rag_config = yaml.safe_load(f)
                nlp_config = rag_config.get('nlp', {})
            except Exception:
                nlp_config = {}
        spacy_model = nlp_config.get('spacy_model', 'en_core_web_sm')
        sentiment_model = nlp_config.get('sentiment_model', 'distilbert-base-uncased-finetuned-sst-2-english')
        self.nlp = spacy.load(spacy_model)
        self.sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _setup_directories(self):
        """Create necessary directories for data collection and processing."""
        insurance_types = ['auto', 'home', 'life', 'health', 'business']
        for dir_type in ['raw', 'processed', 'training']:
            for insurance_type in insurance_types:
                path = self.base_dir / dir_type / insurance_type
                path.mkdir(parents=True, exist_ok=True)
        # Also create the base directories if not present
        (self.base_dir / 'raw').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'processed').mkdir(parents=True, exist_ok=True)
        (self.base_dir / 'training').mkdir(parents=True, exist_ok=True)
    
    def _initialize_apis(self):
        """Initialize API clients."""
        try:
            # Initialize only if credentials are available
            if 'reddit' in self.config and all(k in self.config['reddit'] for k in ['client_id', 'client_secret', 'user_agent']):
                self.reddit = praw.Reddit(
                    client_id=self.config['reddit']['client_id'],
                    client_secret=self.config['reddit']['client_secret'],
                    user_agent=self.config['reddit']['user_agent']
                )
            
            if 'twitter' in self.config and all(k in self.config['twitter'] for k in ['api_key', 'api_secret', 'access_token', 'access_token_secret']):
                auth = tweepy.OAuthHandler(
                    self.config['twitter']['api_key'],
                    self.config['twitter']['api_secret']
                )
                auth.set_access_token(
                    self.config['twitter']['access_token'],
                    self.config['twitter']['access_token_secret']
                )
                self.twitter = tweepy.API(auth)
            
            if 'news_api' in self.config and 'api_key' in self.config['news_api']:
                self.news_api = NewsApiClient(api_key=self.config['news_api']['api_key'])
            
        except Exception as e:
            logger.warning(f"Some APIs could not be initialized: {e}")
    
    async def collect_news_data(self) -> Dict:
        """Collect insurance-related news articles."""
        data = {}
        try:
            news = self.news_api.get_everything(
                q='insurance',
                language='en',
                sort_by='relevancy',
                page_size=100
            )
            
            for article in news['articles']:
                data[article['url']] = {
                    'title': article['title'],
                    'description': article['description'],
                    'content': article['content'],
                    'published_at': article['publishedAt'],
                    'source': article['source']['name']
                }
                
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
        
        return data
    
    async def collect_twitter_data(self) -> Dict:
        """Collect insurance-related tweets."""
        data = {}
        try:
            tweets = self.twitter.search_tweets(
                q="insurance OR #insurance",
                lang="en",
                count=1000,
                tweet_mode="extended"
            )
            
            for tweet in tweets:
                data[tweet.id_str] = {
                    'text': tweet.full_text,
                    'created_at': tweet.created_at,
                    'user': tweet.user.screen_name,
                    'retweet_count': tweet.retweet_count,
                    'favorite_count': tweet.favorite_count
                }
                
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
        
        return data
    
    async def collect_reddit_data(self) -> Dict:
        """Collect insurance-related discussions from Reddit."""
        data = {}
        subreddits = self.config['reddit']['subreddits']
        
        for subreddit in tqdm(subreddits, desc="Collecting Reddit data"):
            try:
                subreddit_instance = self.reddit.subreddit(subreddit['name'])
                posts = subreddit_instance.hot(limit=1000)
                
                data[subreddit['name']] = []
                for post in posts:
                    post_data = {
                        'title': post.title,
                        'text': post.selftext,
                        'score': post.score,
                        'created_utc': post.created_utc,
                        'num_comments': post.num_comments,
                        'url': post.url,
                        'author': str(post.author),
                        'category': subreddit['categories']
                    }
                    data[subreddit['name']].append(post_data)
                    
            except Exception as e:
                logger.error(f"Error collecting data from r/{subreddit}: {e}")
        
        return data
    
    async def collect_insurance_quotes(self) -> Dict:
        """Collect insurance quotes and scenarios from public APIs."""
        data = {}
        async with aiohttp.ClientSession() as session:
            for api in self.config.get('quote_apis', []):
                try:
                    # Add timeout and retry logic
                    for endpoint in api.get('endpoints', []):
                        try:
                            async with session.get(f"{api['base_url']}{endpoint}", timeout=10) as response:
                                if response.status == 200:
                                    quotes = await response.json()
                                    data[f"{api['name']}_{endpoint}"] = quotes
                        except asyncio.TimeoutError:
                            logger.warning(f"Timeout while collecting quotes from {api['name']}{endpoint}")
                        except Exception as e:
                            logger.warning(f"Error collecting quotes from {api['name']}{endpoint}: {e}")
                except Exception as e:
                    logger.warning(f"Error setting up collection from {api['name']}: {e}")
        return data
    
    async def collect_policy_documents(self) -> Dict:
        """Collect sample insurance policy documents."""
        data = {}
        async with aiohttp.ClientSession() as session:
            for source in self.config.get('policy_documents', {}).get('sources', []):
                try:
                    # Add timeout and retry logic
                    try:
                        async with session.get(source['url'], timeout=10) as response:
                            if response.status == 200:
                                html = await response.text()
                                soup = BeautifulSoup(html, 'html.parser')
                                data[source['name']] = {
                                    'content': soup.get_text(),
                                    'categories': source['categories']
                                }
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout while collecting policy documents from {source['name']}")
                    except Exception as e:
                        logger.warning(f"Error collecting policy documents from {source['name']}: {e}")
                except Exception as e:
                    logger.warning(f"Error setting up collection from {source['name']}: {e}")
        return data
    
    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate a conversation entry."""
        try:
            # Check required fields
            if not all(field in conversation for field in self.config['processing']['required_fields']):
                return False
            
            # Check conversation length
            conv_length = len(conversation['conversations'])
            if not (self.config['processing']['min_conversation_length'] <= 
                   conv_length <= 
                   self.config['processing']['max_conversation_length']):
                return False
            
            # Check text length
            for msg in conversation['conversations']:
                if not (self.config['processing']['min_text_length'] <= 
                       len(msg['value']) <= 
                       self.config['processing']['max_text_length']):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating conversation: {e}")
            return False
    
    def process_text(self, text: str) -> str:
        """Process and clean text."""
        try:
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            words = text.split()
            text = ' '.join([word for word in words if word.lower() not in stop_words])
            
            return text
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return text
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        try:
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment
            
            # Transformers sentiment
            transformer_sentiment = self.sentiment_analyzer(text)[0]
            
            return {
                'textblob': {
                    'polarity': textblob_sentiment.polarity,
                    'subjectivity': textblob_sentiment.subjectivity
                },
                'transformer': {
                    'label': transformer_sentiment['label'],
                    'score': transformer_sentiment['score']
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {}
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def process_conversations(self, raw_data: Dict) -> List[Dict]:
        """Process raw data into conversation format."""
        conversations = []
        
        # Process Reddit data
        for subreddit, posts in raw_data.get('reddit', {}).items():
            for post in posts:
                # Process and clean text
                processed_title = self.process_text(post['title'])
                processed_text = self.process_text(post['text'])
                
                # Analyze sentiment
                title_sentiment = self.analyze_sentiment(processed_title)
                text_sentiment = self.analyze_sentiment(processed_text)
                
                # Extract entities
                entities = self.extract_entities(processed_text)
                
                conversation = {
                    'source': f'reddit/r/{subreddit}',
                    'timestamp': datetime.fromtimestamp(post['created_utc']).isoformat(),
                    'metadata': {
                        'score': post['score'],
                        'num_comments': post['num_comments'],
                        'url': post['url'],
                        'author': post['author'],
                        'category': post['category'],
                        'sentiment': {
                            'title': title_sentiment,
                            'text': text_sentiment
                        },
                        'entities': entities
                    },
                    'conversations': [
                        {
                            'from': 'system',
                            'value': 'You are an insurance expert assistant. Provide accurate, helpful information about insurance policies, coverages, and claims.'
                        },
                        {
                            'from': 'human',
                            'value': processed_title
                        },
                        {
                            'from': 'assistant',
                            'value': processed_text
                        }
                    ]
                }
                
                if self.validate_conversation(conversation):
                    conversations.append(conversation)
        
        return conversations
    
    def save_data(self, data: Dict, data_type: str, category: str):
        """Save collected data to JSON file."""
        try:
            # Create output directory
            output_dir = self.base_dir / data_type / category
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"{category}_{timestamp}.json"
            
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(data) if isinstance(data, dict) else len(data)} items to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    async def run(self):
        """Run the data collection process."""
        try:
            # Collect data from alternative sources
            quote_data = await self.collect_insurance_quotes()
            policy_data = await self.collect_policy_documents()
            
            # Save raw data if any was collected
            if quote_data:
                self.save_data(quote_data, 'raw', 'quotes')
            if policy_data:
                self.save_data(policy_data, 'raw', 'policies')
            
            # Process conversations from policy documents
            conversations = []
            for source_name, source_data in policy_data.items():
                try:
                    # Process and clean text
                    processed_text = self.process_text(source_data['content'])
                    
                    # Analyze sentiment
                    text_sentiment = self.analyze_sentiment(processed_text)
                    
                    # Extract entities
                    entities = self.extract_entities(processed_text)
                    
                    conversation = {
                        'source': f'policy_documents/{source_name}',
                        'timestamp': datetime.now().isoformat(),
                        'metadata': {
                            'categories': source_data['categories'],
                            'sentiment': text_sentiment,
                            'entities': entities
                        },
                        'conversations': [
                            {
                                'from': 'system',
                                'value': 'You are an insurance expert assistant. Provide accurate, helpful information about insurance policies, coverages, and claims.'
                            },
                            {
                                'from': 'human',
                                'value': f'What are the key points about {", ".join(source_data["categories"])} insurance from {source_name}?'
                            },
                            {
                                'from': 'assistant',
                                'value': processed_text
                            }
                        ]
                    }
                    
                    if self.validate_conversation(conversation):
                        conversations.append(conversation)
                except Exception as e:
                    logger.error(f"Error processing policy document from {source_name}: {e}")
            
            # Save processed data if any conversations were created
            if conversations:
                self.save_data(conversations, 'processed', 'conversations')
            
        except Exception as e:
            logger.error(f"Error in data collection process: {e}")

if __name__ == "__main__":
    collector = InsuranceDataCollector()
    asyncio.run(collector.run()) 