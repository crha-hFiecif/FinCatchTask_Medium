import pandas as pd
import concurrent.futures
import time
from typing import List, Dict, Optional
import logging
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import spacy
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ArticleExtractor:
    def __init__(self, max_workers: int = 5, rate_limit_delay: float = 0.5):
        """
        Initialize the ArticleExtractor.
        
        Args:
            max_workers: Maximum number of parallel workers
            rate_limit_delay: Delay between requests in seconds
        """
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.nlp = spacy.load('en_core_web_sm')
        
        # Configure session with retries and timeouts
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Set user agent to avoid blocks
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_article(self, url: str, source: str) -> Optional[Dict]:
        """
        Extract article content from a URL.
        
        Args:
            url: The article URL
            source: Source of the article (e.g., 'wiki', 'investopedia')
            
        Returns:
            Dictionary containing title, content, and summary
        """
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            response = self.session.get(url, timeout=10)
            
            # Handle 404 errors gracefully
            if response.status_code == 404:
                logging.warning(f"Article not found (404): {url}")
                return None
                
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            if source.lower() == 'wiki':
                title = soup.find(id='firstHeading')
                if title is None:
                    logging.warning(f"Could not find title for wiki article: {url}")
                    return None
                title = title.text.strip()
                
                content_div = soup.find(id='mw-content-text')
                if content_div is None:
                    logging.warning(f"Could not find content for wiki article: {url}")
                    return None
                    
                paragraphs = content_div.find_all('p')
                
            elif source.lower() == 'investopedia':
                title = soup.find('h1')
                if title is None:
                    logging.warning(f"Could not find title for investopedia article: {url}")
                    return None
                title = title.text.strip()
                
                content_div = soup.find('article')
                if content_div is None:
                    logging.warning(f"Could not find content for investopedia article: {url}")
                    return None
                    
                paragraphs = content_div.find_all('p')
                
            else:
                logging.warning(f"Unsupported source: {source}")
                return None
            
            # Extract meaningful content
            content = '\n'.join(p.text.strip() for p in paragraphs if p.text.strip())
            
            if not content:
                logging.warning(f"No content found for article: {url}")
                return None
            
            # Generate summary using spaCy
            doc = self.nlp(content[:1000])  # Use first 1000 chars for summary
            sentences = list(doc.sents)
            summary = ' '.join(str(sent) for sent in sentences[:2])  # First two sentences
            
            return {
                'title': title,
                'content': content,
                'summary': summary,
                'url': url,
                'source': source
            }
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Error processing {url}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error processing {url}: {str(e)}")
            return None

    def process_csv(self, csv_path: str) -> List[Dict]:
        """
        Process articles from a CSV file in parallel.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of processed articles
        """
        try:
            df = pd.read_csv(csv_path)
            articles = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create future tasks
                future_to_url = {
                    executor.submit(self.extract_article, row['URL'], row['source']): row['URL']
                    for _, row in df.iterrows()
                }
                
                # Process completed tasks with progress bar
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(df)):
                    url = future_to_url[future]
                    try:
                        article = future.result()
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logging.error(f"Error processing {url}: {str(e)}")
            
            logging.info(f"Successfully processed {len(articles)} out of {len(df)} articles")
            return articles
            
        except Exception as e:
            logging.error(f"Error processing CSV file: {str(e)}")
            return []

def main():
    """Main function to demonstrate usage"""
    extractor = ArticleExtractor(max_workers=5, rate_limit_delay=0.5)
    articles = extractor.process_csv('../../data/FinCatch_Sources_Medium.csv')
    
    # Save results
    if articles:
        df_results = pd.DataFrame(articles)
        df_results.to_csv('../../data/processed_articles.csv', index=False)
        logging.info("Results saved to processed_articles.csv")

if __name__ == "__main__":
    main() 