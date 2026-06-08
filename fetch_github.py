import urllib.request
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.in_readme = False

    def handle_starttag(self, tag, attrs):
        if tag == 'article':
            self.in_readme = True

    def handle_endtag(self, tag):
        if tag == 'article':
            self.in_readme = False

    def handle_data(self, data):
        if self.in_readme:
            self.text.append(data.strip())

req = urllib.request.Request("https://github.com/taejunkim/raveform", headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        html = response.read().decode('utf-8')
        parser = MyHTMLParser()
        parser.feed(html)
        print(" ".join([t for t in parser.text if t]))
except Exception as e:
    print(e)
