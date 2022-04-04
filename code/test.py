import requests

# API_URL = "https://rel.cs.ru.nl/api"
API_URL = "http://localhost:5555"
# text_doc = "If you're going to try, go all the way - Charles Bukowski"

# text_doc = "David and Victoria named their children Brooklyn, Romeo, Cruz, and Harper Seven."
text_doc = "Victoria and David added spices on their marriage."

# Example EL.
el_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": []
}).json()

# Example ED.
ed_result = requests.post(API_URL, json={
    "text": text_doc,
    "spans": [(41, 16)]
}).json()

# # Example ED.
# ed_result = requests.post(API_URL, json={
#     "text": text_doc,
#     "spans": [(41, 16)]
# }).json()

print(el_result)
