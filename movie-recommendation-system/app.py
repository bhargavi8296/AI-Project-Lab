import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------------------------
# Load saved artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    loaded_movies = joblib.load("movies.pkl")
    loaded_vectors = joblib.load("tfidf_vectors.pkl")

    # DataFrame positions ko vectors ke saath align rakhega
    loaded_movies = loaded_movies.reset_index(drop=True)

    return loaded_movies, loaded_vectors


try:
    movies, tfidf_vectors = load_artifacts()

except FileNotFoundError as error:
    st.error(
        "Required files nahi mili. Ensure karo ki "
        "movies.pkl aur tfidf_vectors.pkl, app.py ke "
        "same folder mein present hain."
    )
    st.exception(error)
    st.stop()

except Exception as error:
    st.error("Saved model artifacts load nahi ho paaye.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Validate artifacts
# --------------------------------------------------
if len(movies) != tfidf_vectors.shape[0]:
    st.error(
        "Artifact mismatch: movies aur TF-IDF vectors "
        "ki rows equal nahi hain."
    )

    st.write("Total movies:", len(movies))
    st.write("Total vector rows:", tfidf_vectors.shape[0])

    st.stop()


required_columns = {"movie_id", "title"}

if not required_columns.issubset(movies.columns):
    st.error(
        "movies.pkl mein movie_id aur title columns "
        "available nahi hain."
    )
    st.stop()


# --------------------------------------------------
# Recommendation function
# --------------------------------------------------
def recommend(movie_title, top_n=5):
    matching_movies = movies[
        movies["title"] == movie_title
    ]

    if matching_movies.empty:
        return []

    movie_index = matching_movies.index[0]

    # Slicing ki wajah se shape (1, 5000) rahegi
    selected_movie_vector = tfidf_vectors[
        movie_index:movie_index + 1
    ]

    similarity_scores = cosine_similarity(
        selected_movie_vector,
        tfidf_vectors
    ).flatten()

    # Highest similarity score se lowest ki taraf
    sorted_indices = similarity_scores.argsort()[::-1]

    recommendations = []

    for index in sorted_indices:

        # Selected movie ko khud recommend nahi karna
        if index == movie_index:
            continue

        recommendations.append(
            {
                "movie_id": int(
                    movies.iloc[index]["movie_id"]
                ),
                "title": movies.iloc[index]["title"],
                "score": float(
                    similarity_scores[index]
                )
            }
        )

        if len(recommendations) == top_n:
            break

    return recommendations


# --------------------------------------------------
# Application heading
# --------------------------------------------------
st.title("🎬 Movie Recommendation System")

st.write(
    """
    Apni pasand ki movie select karo aur content-based
    filtering ki help se similar movies discover karo.
    """
)

st.caption(
    """
    Recommendations movie overview, genres, keywords,
    cast aur director ke basis par generate hoti hain.
    """
)

st.divider()


# --------------------------------------------------
# User inputs
# --------------------------------------------------
movie_titles = sorted(
    movies["title"]
    .dropna()
    .unique()
    .tolist()
)

selected_movie = st.selectbox(
    label="Choose a movie",
    options=movie_titles,
    index=None,
    placeholder="Search or select a movie"
)

number_of_recommendations = st.slider(
    label="Number of recommendations",
    min_value=3,
    max_value=10,
    value=5,
    step=1
)

recommend_button = st.button(
    label="Get Recommendations",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Generate and display recommendations
# --------------------------------------------------
if recommend_button:

    if selected_movie is None:
        st.warning("Please pehle koi movie select karo.")

    else:
        with st.spinner(
            "Similar movies find ki ja rahi hain..."
        ):
            recommendations = recommend(
                selected_movie,
                number_of_recommendations
            )

        if not recommendations:
            st.warning(
                "Selected movie ke liye recommendations "
                "nahi mili."
            )

        else:
            st.success(
                f"Recommendations based on: {selected_movie}"
            )

            st.subheader("Movies you may like")

            # Ek row mein 3 recommendation cards
            columns = st.columns(3)

            for position, movie in enumerate(
                recommendations
            ):
                column = columns[position % 3]

                with column:
                    with st.container(border=True):

                        st.caption(
                            f"RECOMMENDATION {position + 1}"
                        )

                        st.subheader(
                            movie["title"]
                        )

                        st.metric(
                            label="Similarity score",
                            value=f'{movie["score"]:.3f}'
                        )


            # Recommendation system explanation
            with st.expander(
                "How does this recommendation system work?"
            ):
                st.write(
                    """
                    First, every movie's overview, genres,
                    keywords, top cast members and director
                    are combined into a single tags column.
                    """
                )

                st.write(
                    """
                    TF-IDF converts these textual tags into
                    numerical vectors. Cosine similarity then
                    compares the selected movie's vector with
                    all other movie vectors.
                    """
                )

                st.info(
                    """
                    Similarity score content overlap represent
                    karta hai. Ye accuracy percentage, movie
                    rating ya success probability nahi hai.
                    """
                )


# --------------------------------------------------
# Model information
# --------------------------------------------------
st.divider()

with st.expander("Project details"):
    st.write("**Recommendation type:** Content-based filtering")
    st.write("**Text representation:** TF-IDF")
    st.write("**Similarity method:** Cosine similarity")
    st.write(f"**Movies available:** {len(movies):,}")
    st.write(f"**TF-IDF features:** {tfidf_vectors.shape[1]:,}")


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.caption(
    "Built with Python, Pandas, Scikit-learn and Streamlit"
)