import requests
from langchain.tools import tool

@tool
def search_games(query: str) -> str:
    """
    Searches the Steam store for a game using a text query.
    Returns a list of matching games with their App IDs and prices.
    """
    url = f"https://store.steampowered.com/api/storesearch/?term={query}&l=english&cc=US"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Failed to fetch data from Steam API."
        
    data = response.json()
    if not data.get("total", 0):
        return f"No games found for '{query}'."
        
    results = []
    for item in data.get("items", [])[:5]:
        name = item.get("name")
        appid = item.get("id")
        price = item.get("price", {}).get("final", 0) / 100 # from cents to dollas
        results.append(f"Name: {name} | AppID: {appid} | Price: ${price}")
        
    return "\n".join(results)

@tool
def get_game_details(appid: int) -> str:
    """
    Fetches detailed information about a specific Steam game using its App ID.
    Returns the game's description, genres, and review score.
    """
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"
    response = requests.get(url)
    
    if response.status_code != 200:
        return "Failed to fetch game details."
        
    data = response.json()
    if not data.get(str(appid), {}).get("success"):
        return f"Could not find details for AppID {appid}."
        
    game_data = data[str(appid)]["data"]
    name = game_data.get("name", "Unknown")
    description = game_data.get("short_description", "No description available.")
    genres = ", ".join([g["description"] for g in game_data.get("genres", [])])
    
    return f"Game: {name}\nGenres: {genres}\nDescription: {description}"


@tool
def get_game_reviews(appid: int) -> str:
    """
    Fetches recent reviews for a specific steam game using the app id.
    """
    url = f"https://store.steampowered.com/appreviews/{appid}?json=1&language=all&filter=recent&num_per_page=3"
    response = requests.get(url)

    if response.status_code != 200:
        return "Failed to fetch game reviews"

    data = response.json()
    if data.get("success") != 1 or not data.get("reviews"):
        return f"No Reviews found for AppID {appid}."

    summary = data.get("query_summary", {})
    review_score = summary.get("review_score_desc", "Unknown")
    total_reviews = summary.get("total_reviews", 0)

    reviews_plain_text = []
    for review in data["reviews"]:
        text = review.get("review", "").replace('\n', ' ')
        vote = "upvote" if review.get("voted_up") else "downvote"

        reviews_plain_text.append(f"{vote}: {text}")

    return f"Overall Rating: {review_score} ({total_reviews} reviews)\n\nRecent Player Reviews:\n{"\n".join(reviews_plain_text)}"
