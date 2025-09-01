
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import asyncio
import aiohttp
import schedule
import time
import threading
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    import wikipedia
    import feedparser
    from bs4 import BeautifulSoup
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Auto-learning dependencies not installed. Install with: pip install wikipedia feedparser beautifulsoup4 aiohttp schedule")

class ContentProcessor:
    """Process and filter content for quality and relevance."""
    
    def __init__(self):
        self.min_content_length = 100
        self.max_content_length = 5000
        self.quality_keywords = [
            'research', 'study', 'analysis', 'theory', 'method', 'algorithm',
            'science', 'technology', 'mathematics', 'physics', 'chemistry',
            'biology', 'engineering', 'computer', 'artificial', 'intelligence'
        ]
    
    def is_quality_content(self, content: str) -> bool:
        """Determine if content meets quality standards."""
        if not content or len(content) < self.min_content_length:
            return False
        
        # Check for educational/scientific keywords
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in self.quality_keywords if keyword in content_lower)
        
        # Content should have at least some educational keywords
        return keyword_count >= 2
    
    def extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content."""
        # Simple concept extraction using capitalized terms and keywords
        concepts = []
        
        # Find capitalized terms (potential concepts)
        capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        concepts.extend(capitalized_terms[:10])  # Limit to top 10
        
        # Add domain-specific terms
        domain_terms = [
            'machine learning', 'deep learning', 'neural network', 'algorithm',
            'artificial intelligence', 'data science', 'programming', 'mathematics',
            'physics', 'chemistry', 'biology', 'engineering', 'research'
        ]
        
        content_lower = content.lower()
        for term in domain_terms:
            if term in content_lower and term not in [c.lower() for c in concepts]:
                concepts.append(term.title())
        
        return concepts[:15]  # Return top 15 concepts
    
    def score_importance(self, content: str, source: str) -> float:
        """Score content importance based on various factors."""
        score = 0.5  # Base score
        
        content_lower = content.lower()
        
        # Source credibility
        if 'wikipedia' in source:
            score += 0.2
        elif 'arxiv' in source:
            score += 0.3
        elif any(domain in source for domain in ['edu', 'gov', 'org']):
            score += 0.15
        
        # Content quality indicators
        if 'research' in content_lower:
            score += 0.1
        if 'study' in content_lower:
            score += 0.1
        if re.search(r'\d{4}', content):  # Contains years (research papers)
            score += 0.05
        if len(content) > 1000:  # Substantial content
            score += 0.1
        
        return min(score, 1.0)


class KnowledgeSource:
    """Base class for knowledge sources."""
    
    def __init__(self, name: str):
        self.name = name
        self.last_update = None
    
    async def fetch_content(self) -> List[Dict[str, Any]]:
        """Fetch content from the source."""
        raise NotImplementedError


class WikipediaSource(KnowledgeSource):
    """Wikipedia knowledge source."""
    
    def __init__(self):
        super().__init__("Wikipedia")
        self.topics = [
            "Artificial intelligence", "Machine learning", "Deep learning",
            "Computer science", "Mathematics", "Physics", "Chemistry",
            "Biology", "Engineering", "Data science", "Programming",
            "Quantum computing", "Robotics", "Neuroscience", "Statistics"
        ]
    
    async def fetch_content(self) -> List[Dict[str, Any]]:
        """Fetch Wikipedia articles."""
        if not HAS_DEPS:
            return []
        
        content_list = []
        
        for topic in self.topics:
            try:
                # Get summary and full page
                summary = wikipedia.summary(topic, sentences=5)
                page = wikipedia.page(topic)
                
                content_list.append({
                    "title": page.title,
                    "content": f"{summary}\n\n{page.content[:2000]}",
                    "source": f"wikipedia.org/wiki/{page.title.replace(' ', '_')}",
                    "type": "encyclopedia",
                    "timestamp": datetime.now().isoformat()
                })
                
                await asyncio.sleep(1)  # Rate limiting
                
            except Exception as e:
                logging.warning(f"Failed to fetch Wikipedia topic {topic}: {e}")
                continue
        
        return content_list


class RSSFeedSource(KnowledgeSource):
    """RSS feed knowledge source."""
    
    def __init__(self):
        super().__init__("RSS Feeds")
        self.feeds = [
            "https://rss.cnn.com/rss/edition.rss",
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
            "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml",
            "https://feeds.feedburner.com/oreilly/radar/data",
            "https://techcrunch.com/feed/",
        ]
    
    async def fetch_content(self) -> List[Dict[str, Any]]:
        """Fetch RSS feed articles."""
        if not HAS_DEPS:
            return []
        
        content_list = []
        
        async with aiohttp.ClientSession() as session:
            for feed_url in self.feeds:
                try:
                    async with session.get(feed_url) as response:
                        if response.status == 200:
                            feed_data = await response.text()
                            feed = feedparser.parse(feed_data)
                            
                            for entry in feed.entries[:5]:  # Limit to 5 per feed
                                content_list.append({
                                    "title": entry.title,
                                    "content": getattr(entry, 'summary', entry.title),
                                    "source": entry.link,
                                    "type": "news",
                                    "timestamp": datetime.now().isoformat()
                                })
                    
                    await asyncio.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    logging.warning(f"Failed to fetch RSS feed {feed_url}: {e}")
                    continue
        
        return content_list


class ArxivSource(KnowledgeSource):
    """ArXiv papers knowledge source."""
    
    def __init__(self):
        super().__init__("ArXiv")
        self.categories = ["cs.AI", "cs.LG", "cs.CL", "stat.ML", "math.ST"]
    
    async def fetch_content(self) -> List[Dict[str, Any]]:
        """Fetch ArXiv papers."""
        content_list = []
        
        async with aiohttp.ClientSession() as session:
            for category in self.categories:
                try:
                    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending"
                    
                    async with session.get(url) as response:
                        if response.status == 200:
                            xml_data = await response.text()
                            # Simple XML parsing for ArXiv
                            entries = re.findall(r'<entry>(.*?)</entry>', xml_data, re.DOTALL)
                            
                            for entry in entries[:5]:  # Limit to 5 per category
                                title_match = re.search(r'<title>(.*?)</title>', entry)
                                summary_match = re.search(r'<summary>(.*?)</summary>', entry)
                                
                                if title_match and summary_match:
                                    content_list.append({
                                        "title": title_match.group(1).strip(),
                                        "content": summary_match.group(1).strip()[:1000],
                                        "source": f"arxiv.org/{category}",
                                        "type": "research",
                                        "timestamp": datetime.now().isoformat()
                                    })
                    
                    await asyncio.sleep(3)  # Rate limiting
                    
                except Exception as e:
                    logging.warning(f"Failed to fetch ArXiv category {category}: {e}")
                    continue
        
        return content_list


class AutoLearningSystem:
    """Main auto-learning system that coordinates knowledge acquisition."""
    
    def __init__(self, memory_layer, storage_dir: str = "auto_learning"):
        self.memory_layer = memory_layer
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.content_processor = ContentProcessor()
        self.sources = [
            WikipediaSource(),
            RSSFeedSource(),
            ArxivSource()
        ]
        
        self.learning_stats = {
            "total_items_learned": 0,
            "last_learning_cycle": None,
            "learning_topics": set(),
            "source_stats": {},
            "learning_history": []
        }
        
        self.is_learning = False
        self.learning_thread = None
        
        # Load existing stats
        self._load_stats()
        
        # Setup learning schedule
        self._setup_schedule()
    
    def _setup_schedule(self):
        """Setup learning schedule."""
        # Full learning cycle every 7 days
        schedule.every(7).days.do(self._scheduled_full_learning)
        
        # Quick updates every hour
        schedule.every().hour.do(self._scheduled_quick_learning)
        
        # Start scheduler thread
        self.learning_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.learning_thread.start()
    
    def _run_scheduler(self):
        """Run the learning scheduler."""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _scheduled_full_learning(self):
        """Scheduled full learning cycle."""
        if not self.is_learning:
            asyncio.create_task(self.learn_from_all_sources())
    
    def _scheduled_quick_learning(self):
        """Scheduled quick learning update."""
        if not self.is_learning:
            asyncio.create_task(self.quick_learning_update())
    
    async def learn_from_all_sources(self) -> Dict[str, Any]:
        """Learn from all configured sources."""
        if self.is_learning:
            return {"status": "already_learning"}
        
        self.is_learning = True
        learning_start = datetime.now()
        
        try:
            all_content = []
            source_stats = {}
            
            # Fetch from all sources
            for source in self.sources:
                try:
                    content = await source.fetch_content()
                    all_content.extend(content)
                    source_stats[source.name] = len(content)
                    source.last_update = datetime.now()
                except Exception as e:
                    logging.error(f"Error learning from {source.name}: {e}")
                    source_stats[source.name] = 0
            
            # Process and filter content
            processed_content = []
            for item in all_content:
                if self.content_processor.is_quality_content(item["content"]):
                    # Add metadata
                    item["concepts"] = self.content_processor.extract_concepts(item["content"])
                    item["importance_score"] = self.content_processor.score_importance(
                        item["content"], item["source"]
                    )
                    processed_content.append(item)
            
            # Add to memory layer
            documents = []
            for item in processed_content:
                doc = f"Title: {item['title']}\nSource: {item['source']}\nConcepts: {', '.join(item['concepts'])}\nContent: {item['content']}"
                documents.append(doc)
            
            if documents:
                self.memory_layer.add_knowledge_documents(documents)
            
            # Update stats
            self.learning_stats["total_items_learned"] += len(processed_content)
            self.learning_stats["last_learning_cycle"] = learning_start.isoformat()
            self.learning_stats["source_stats"] = source_stats
            
            for item in processed_content:
                self.learning_stats["learning_topics"].update(item["concepts"])
            
            # Record learning session
            session_record = {
                "timestamp": learning_start.isoformat(),
                "items_learned": len(processed_content),
                "sources_used": list(source_stats.keys()),
                "duration_seconds": (datetime.now() - learning_start).total_seconds()
            }
            self.learning_stats["learning_history"].append(session_record)
            
            # Keep only last 50 history records
            if len(self.learning_stats["learning_history"]) > 50:
                self.learning_stats["learning_history"] = self.learning_stats["learning_history"][-50:]
            
            self._save_stats()
            
            return {
                "status": "success",
                "items_learned": len(processed_content),
                "sources_processed": len(self.sources),
                "duration_seconds": (datetime.now() - learning_start).total_seconds()
            }
            
        finally:
            self.is_learning = False
    
    async def quick_learning_update(self) -> Dict[str, Any]:
        """Quick learning update from fast sources."""
        if self.is_learning:
            return {"status": "already_learning"}
        
        self.is_learning = True
        
        try:
            # Only use RSS feeds for quick updates
            rss_source = next((s for s in self.sources if isinstance(s, RSSFeedSource)), None)
            if not rss_source:
                return {"status": "no_quick_sources"}
            
            content = await rss_source.fetch_content()
            
            # Process content
            processed_content = []
            for item in content:
                if self.content_processor.is_quality_content(item["content"]):
                    item["concepts"] = self.content_processor.extract_concepts(item["content"])
                    processed_content.append(item)
            
            # Add to memory
            if processed_content:
                documents = [f"Title: {item['title']}\nContent: {item['content']}" for item in processed_content]
                self.memory_layer.add_knowledge_documents(documents)
                self.learning_stats["total_items_learned"] += len(processed_content)
            
            return {
                "status": "success",
                "items_learned": len(processed_content),
                "type": "quick_update"
            }
            
        finally:
            self.is_learning = False
    
    async def learn_from_url(self, url: str) -> Dict[str, Any]:
        """Learn from a specific URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser') if HAS_DEPS else None
                        
                        if soup:
                            # Extract text content
                            for script in soup(["script", "style"]):
                                script.decompose()
                            text = soup.get_text()
                            
                            # Clean text
                            lines = (line.strip() for line in text.splitlines())
                            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                            text = ' '.join(chunk for chunk in chunks if chunk)
                            
                            if self.content_processor.is_quality_content(text):
                                document = f"URL: {url}\nContent: {text[:2000]}"
                                self.memory_layer.add_knowledge_documents([document])
                                
                                return {
                                    "status": "success",
                                    "url": url,
                                    "content_length": len(text)
                                }
                        
                        return {"status": "failed", "reason": "Could not extract quality content"}
                    else:
                        return {"status": "failed", "reason": f"HTTP {response.status}"}
        
        except Exception as e:
            return {"status": "failed", "reason": str(e)}
    
    def add_learning_topic(self, topic: str):
        """Add a new topic to learn about."""
        # Add to Wikipedia source topics
        wiki_source = next((s for s in self.sources if isinstance(s, WikipediaSource)), None)
        if wiki_source and topic not in wiki_source.topics:
            wiki_source.topics.append(topic)
            self.learning_stats["learning_topics"].add(topic)
            self._save_stats()
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status."""
        return {
            "is_learning": self.is_learning,
            "total_items_learned": self.learning_stats["total_items_learned"],
            "last_learning_cycle": self.learning_stats["last_learning_cycle"],
            "learning_topics_count": len(self.learning_stats["learning_topics"]),
            "sources_count": len(self.sources),
            "recent_history": self.learning_stats["learning_history"][-5:] if self.learning_stats["learning_history"] else []
        }
    
    def _save_stats(self):
        """Save learning statistics."""
        stats_file = self.storage_dir / "learning_stats.json"
        
        # Convert sets to lists for JSON serialization
        stats_to_save = self.learning_stats.copy()
        stats_to_save["learning_topics"] = list(stats_to_save["learning_topics"])
        
        with open(stats_file, 'w') as f:
            json.dump(stats_to_save, f, indent=2)
    
    def _load_stats(self):
        """Load learning statistics."""
        stats_file = self.storage_dir / "learning_stats.json"
        
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    loaded_stats = json.load(f)
                
                # Convert lists back to sets
                if "learning_topics" in loaded_stats:
                    loaded_stats["learning_topics"] = set(loaded_stats["learning_topics"])
                
                self.learning_stats.update(loaded_stats)
            except Exception as e:
                logging.warning(f"Failed to load learning stats: {e}")
