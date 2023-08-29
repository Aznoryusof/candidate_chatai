import os
import sys

import webbrowser
from dotenv import load_dotenv
from ansi2html import Ansi2HTMLConverter

load_dotenv()

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)
LOG_FILE_PATH = os.path.join(MAIN_DIR, os.environ.get('LOG_FILE_PATH'))

with open(LOG_FILE_PATH, "r") as f:
    content = f.readlines()

content_list = list(dict.fromkeys(content))
content = "".join(content_list)

conv = Ansi2HTMLConverter()
html = conv.convert(content, full=True)

with open(LOG_FILE_PATH+".html", "w") as file:
    file.write(html)

webbrowser.open(LOG_FILE_PATH+".html")