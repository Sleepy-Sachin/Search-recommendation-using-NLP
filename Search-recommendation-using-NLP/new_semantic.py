import logging
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer, util

# Initialize logging
logging.basicConfig(level=logging.INFO)

# MongoDB connection setup
client = MongoClient('mongodb://localhost:27017/')
db = client['video_database']
video_collection = db['video_postives']
query_matches_collection = db['Semantic']

# Load pre-trained SBERT model
model = SentenceTransformer('paraphrase-MPNet-base-v2')

# Fetch video data from MongoDB
try:
    videos = list(video_collection.find({}, {'_id': 1, 'title': 1, 'description': 1}))
    logging.info(f"Fetched {len(videos)} videos from MongoDB.")
except Exception as e:
    logging.error(f"Error fetching videos from MongoDB: {e}")
    videos = []

# Define the search queries
food_search_queries = [
    "Authentic dosa recipe",
    "Best Mumbai street food",
    "Healthy Indian breakfast ideas",
    "Hyderabadi biryani recipe",
    "Paneer curry",
    "Homemade samosa",
    "Masala chai",
    "Spicy mango pickle",
    "Gulab jamun recipe",
    "Vegetarian curry",
    "Naan bread",
    "Chicken tikka masala",
    "Beef biryani recipe",
    "Sambar",
    "Vegetable pakoras",
    "Indian street food recipe"
]

# Combine title and description for semantic analysis, giving title more weight
video_texts = [f"{video['title']} {video['title']} {video['description']}" for video in videos]

# Encode video data into embeddings using SBERT
video_embeddings = model.encode(video_texts, convert_to_tensor=True, show_progress_bar=True)
logging.info("Encoded video texts into embeddings.")

# Process each food search query without expansion
for query in food_search_queries:
    logging.info(f"Processing query: {query}")

    # Encode the query into embeddings
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity between the query and video embeddings
    cosine_scores = util.pytorch_cos_sim(query_embedding, video_embeddings)[0]

    # Lowered threshold for testing
    relevant_video_indices = (cosine_scores >= 0.58).nonzero(as_tuple=True)[0]

    # Collect video IDs and titles for relevant matches
    relevant_videos = [{"id": videos[idx]['_id'], "title": videos[idx]['title']} for idx in relevant_video_indices]

    # If relevant videos are found, store them in MongoDB
    if relevant_videos:
        query_data = {
            'search_query': query,
            'videos': relevant_videos  # Store both ID and title
        }
        try:
            result = query_matches_collection.update_one(
                {'search_query': query},
                {'$set': query_data},
                upsert=True
            )
            if result.upserted_id:
                logging.info(f"Inserted new document for query: '{query}' with ID: {result.upserted_id}")
            else:
                logging.info(f"Updated existing document for query: '{query}'")
            logging.info(f"Stored {len(relevant_videos)} relevant videos for query: '{query}'")
        except Exception as e:
            logging.error(f"Error storing data for query '{query}': {e}")
    else:
        logging.info(f"No relevant videos found for query: '{query}'")

print("Query and relevant video matches stored successfully.")
