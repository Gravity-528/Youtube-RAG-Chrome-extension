import scrapy
from urllib.parse import urlparse
from scrapy.exceptions import CloseSpider
from scrapy.crawler import CrawlerProcess

class RecursiveInMemorySpider(scrapy.Spider):
    name = "documentation"
    visited = set()
    scraped_docs = []
    max_pages = 5

    # custom_settings = {
    #     "REDIRECT_ENABLED": False  # Optional: block 301/302 redirects to offsite domains
    # }

    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not start_url:
            raise ValueError("You must provide a start_url")
        
        self.start_url = start_url.strip()
        self.base_domain = urlparse(self.start_url).netloc
        self.allowed_domains = [self.base_domain]

    def start(self):
        yield scrapy.Request(url=self.start_url, callback=self.parse)

    def parse(self, response):
        if response.url in self.visited:
            return
        self.visited.add(response.url)

        texts = response.xpath(
            '//body//text()[not(ancestor::script or ancestor::style or ancestor::a)]'
        ).getall()
        cleaned = [t.strip() for t in texts if t.strip()]
        if not cleaned:
            return

        full_text = "\n".join(cleaned)

        # self.scraped_docs.append({
        #     "url": response.url,
        #     "text": full_text,
        #     "source": self.base_domain
        # })
        item = {
           "url": response.url,
            "text": full_text,
            "source": self.base_domain
    }
        self.scraped_docs.append(item)
        yield item

        self.log(f"[✓] Scraped: {response.url} — total: {len(self.scraped_docs)}")

        for href in response.css("a::attr(href)").getall():
            next_url = response.urljoin(href)
            if self.is_internal(next_url) and next_url not in self.visited:
                yield scrapy.Request(url=next_url, callback=self.parse)

        if len(self.visited) >= self.max_pages:
            raise CloseSpider("Reached page limit")

    def is_internal(self, url):
        return urlparse(url).netloc == self.base_domain
    
    def get_scraped_docs(self):
        return self.scraped_docs

process = CrawlerProcess(
    settings={
        "FEEDS": {
            "items.json": {"format": "json"},
        },
    }
)

# process.crawl(RecursiveInMemorySpider)
# process.start()