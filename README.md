# Movie Sage Recommender

Movie Sage Recommender is an advanced movie recommendation system built with Python and Streamlit. It uses natural language processing and content-based filtering to suggest movies based on user input.

## Features

- Content-based movie recommendations
- User-friendly Streamlit interface
- Efficient model persistence for quick loading
- Customizable number of recommendations

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/movie-sage-recommender.git
   cd movie-sage-recommender
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the MovieLens dataset:
   - Visit [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)
   - Download the desired dataset size (e.g., "MovieLens 100K Dataset" or "MovieLens 1M Dataset")
   - Extract the downloaded file and place `movies.csv` , `ratings.csv` and `ratings.csv` in the project directory

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`)

3. Enter a movie description or title in the text area and get personalized movie recommendations!

## How it Works

Movie Sage Recommender uses a sentence transformer model to encode movie descriptions and user inputs into numerical vectors. It then calculates the similarity between these vectors to find the most relevant movie recommendations.

## Contributing

Contributions to Movie Sage Recommender are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
