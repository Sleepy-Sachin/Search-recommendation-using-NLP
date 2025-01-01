from flask import Flask, render_template, request, redirect, url_for, flash
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import os  # Import os for generating a random secret key

app = Flask(__name__)

# Set a secret key for session management
app.secret_key = os.urandom(24)  # Generate a random secret key

# Configure MongoDB
app.config["MONGO_URI"] = "mongodb://localhost:27017/video_database"  # Update with your MongoDB connection string
mongo = PyMongo(app)

@app.route('/')
def home():
    # Fetch video data from the video_postives collection
    videos = mongo.db.video_postives.find()
    return render_template('home.html', videos=videos)

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword', '').strip()
    if not keyword:
        flash("Please enter a search query.")
        return redirect(url_for('home'))

    # Semantic search
    semantic_result = mongo.db.Semantic.find_one({'search_query': keyword})

    if semantic_result and 'videos' in semantic_result:
        matching_video_ids = [video['id'] for video in semantic_result['videos']]
        try:
            # Fetch videos using their IDs, sorted by likes and views
            videos = mongo.db.video_postives.find({
                '_id': {'$in': [ObjectId(vid) for vid in matching_video_ids]}
            }).sort([('likes', -1), ('views', -1)])  # Sort by likes and views in descending order

            video_list = list(videos)  # Debugging line
            if video_list:
                print('Semantic Search')
                return render_template('search_results.html', keyword=keyword, videos=video_list)
        except Exception as e:
            print(f"Error fetching videos: {e}")

    # Fallback search by title
    videos = mongo.db.video_postives.find({
        '$or': [
            {'title': {'$regex': keyword, '$options': 'i'}}
        ]
    }).sort([('likes', -1), ('views', -1)])  # Sort by likes and views in descending order

    video_list = list(videos)  # Debugging line
    if not video_list:
        print('Re Search')
        return render_template('search_results.html', keyword=keyword, videos=None)

    return render_template('search_results.html', keyword=keyword, videos=video_list)

@app.route('/like/<video_id>', methods=['POST'])
def like_video(video_id):
    mongo.db.video_postives.update_one({"_id": ObjectId(video_id)}, {"$inc": {"likes": 1}})
    return redirect(request.referrer)

@app.route('/comment/<video_id>', methods=['POST'])
def comment_video(video_id):
    comment = request.form.get('comment', '').strip()
    if comment and len(comment) <= 200:  # Limiting comment length
        mongo.db.video_postives.update_one(
            {"_id": ObjectId(video_id)},
            {"$push": {"comments": comment}}
        )
    return redirect(request.referrer)

if __name__ == '__main__':
    app.run(debug=True)
