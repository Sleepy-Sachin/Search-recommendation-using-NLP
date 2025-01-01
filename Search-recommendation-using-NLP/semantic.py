from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch


class SemanticAnalyzer:
    def __init__(self, db_name='video_database', collection_name='video_postives'):
        # Initialize MongoDB client and collections
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.semantic_collection = self.db['Semantic']

        # Load SBERT model for embedding generation
        self.model = SentenceTransformer(
            'paraphrase-MiniLM-L6-v2')  # You can also use a larger model like 'paraphrase-MPNet-base-v2' for better accuracy

        # Load all video documents from MongoDB and create a corpus for BM25
        self.all_videos = list(self.collection.find())
        self.corpus = [video['title'] + " " + video['description'] for video in self.all_videos]
        tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)  # Initialize BM25 with tokenized documents

    def get_embeddings(self, text):
        """Generate SBERT embeddings for the given text."""
        if not text:
            return None
        return self.model.encode([text], convert_to_tensor=True)

    def bm25_search(self, query, top_n=50):
        """Use BM25 to find top N relevant videos based on keyword matching."""
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_videos = sorted(zip(bm25_scores, self.all_videos), key=lambda x: x[0], reverse=True)
        return [video for score, video in top_videos[:top_n]]  # Return top N results

    def analyze_video_content(self, search_query):
        """Perform both BM25-based keyword search and SBERT-based semantic ranking."""
        if not search_query:
            return []

        # Step 1: BM25 Keyword Filtering
        filtered_videos = self.bm25_search(search_query)

        # Step 2: Semantic Re-ranking using SBERT
        query_embedding = self.get_embeddings(search_query)
        matching_videos = []

        for video in filtered_videos:
            title_embedding = self.get_embeddings(video['title'])
            description_embedding = self.get_embeddings(video['description'])

            # Compute cosine similarities
            if title_embedding is not None and description_embedding is not None:
                title_similarity = torch.nn.functional.cosine_similarity(query_embedding, title_embedding)
                description_similarity = torch.nn.functional.cosine_similarity(query_embedding, description_embedding)

                # Combine title and description similarity with weighting (90% title, 10% description)
                combined_similarity = 0.7 * title_similarity.item() + 0.3 * description_similarity.item()

                # Threshold for determining if video is a semantic match (adjust based on performance)
                if combined_similarity > 0.41:
                    matching_videos.append(video)

        # Optional: Store the search query and matching video ids in the Semantic collection
        video_ids = [video['_id'] for video in matching_videos]
        titles = [video['title'] for video in matching_videos]
        self.semantic_collection.insert_one({
            'search_query': search_query,
            'matching_video_ids': video_ids,
            'titles': titles
        })

        return matching_videos


# Define the list of search queries related to Indian food
food_search_queries = [
    "Authentic dosa recipe.",
    "Best Mumbai street food.",
    "Healthy Indian breakfast ideas.",
    "Hyderabadi biryani recipe.",
    "paneer curry.",
    "Homemade samosa.",
    "masala chai.",
    "Spicy mango pickle.",
    "Gulab jamun recipe.",
    "vegetarian curry.",
    "naan bread.",
    "Chicken tikka masala.",
    "beef biryani recipe.",
    "sambar.",
    "Vegetable pakoras.",
    "Indian street food recipe."
]


# Main function to perform semantic analysis on the food-related search queries
def main():
    analyzer = SemanticAnalyzer()

    for query in food_search_queries:
        print(query,"Searching")
        matching_videos = analyzer.analyze_video_content(query)
        print(f"Processed query: {query}")
        print(f"Matching videos: {[video['title'] for video in matching_videos]}")


if __name__ == "__main__":
    main()
