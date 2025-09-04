from scrapy import Spider


class BooksToScrapeComSpider(Spider):
    name = "books_toscrape_com"
    custom_settings = {
        "CONCURRENT_REQUESTS_PER_DOMAIN": 8,
        "DOWNLOAD_DELAY": 0.01,
    }
    start_urls = [
        "http://books.toscrape.com/catalogue/page-3.html"
    ]

    def parse(self, response):
        next_page_links = response.css(".next a")
        yield from response.follow_all(next_page_links)
        book_links = response.css("article a")
        yield from response.follow_all(book_links, callback=self.parse_book)

    def parse_book(self, response):
        yield {
            "name": response.css("h1::text").get(),
            "Description": response.css("div#product_description ~ p::text").get(),
            "price": response.css(".price_color::text").re_first("£(.*)"),
            "stars": response.css("p.star-rating::attr(class)").re_first(r"\bstar-rating\s+(\w+)\b"),
            "url": response.url,
        }