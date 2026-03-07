import time
from semantic_cache import process_query, SIMILARITY_THRESHOLD

queries = [
    "How do rockets work?",
    "Explain rocket engines",
    "Why do earthquakes happen?",
    "What causes earthquakes?",
    "How does NASA launch rockets?",
    "Explain space propulsion",
    "Why does the earth shake?",
    "What triggers earthquakes?"
]

hits = 0
misses = 0

start_time = time.time()

for q in queries:

    print("\nQuery:", q)

    before = time.time()

    result = process_query(q)

    after = time.time()

    duration = after - before

    print("Time taken:", round(duration,3),"seconds")

end_time = time.time()

print("\nTotal Runtime:", round(end_time-start_time,3),"seconds")