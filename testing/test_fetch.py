import urllib.request
import re

html = urllib.request.urlopen('https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx?RaceDate=2023/01/01').read().decode('utf-8')
print(re.findall(r'class="([^"]*table[^"]*)"', html))
