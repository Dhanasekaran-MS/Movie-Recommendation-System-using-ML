# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:06:21 2024

@author: dhana
"""

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from streamlit_option_menu import option_menu

vectorizer = TfidfVectorizer()

movies_data = pd.read_csv("movies.csv")
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')
    
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

list_of_all_movies = movies_data['title'].tolist()
            
# Streamlit Interface
with st.sidebar:
    selected = option_menu('Movie Recommendation',
                           ['About',
                            'Recommendation'],
                           icons=['house', 'film'],
                           default_index=0)
if selected == 'About':
    st.title('Movie Recommendation System')
    st.header('About Project :')
    st.markdown('- This Project is Aimed to recommend 10 Movies')
    st.markdown('- Movies will be Recommended Based on a user given movie')
    st.markdown('- The System Will suggest the Movie Based on the similarity Score')
    st.markdown('- The similarity Score will be calculated based on the genre, Director etc..')
    st.header('How To Use :')
    st.markdown('- Select Recommendation Option Menu ')
    st.markdown('- Give an movie name as a input in the textbox')
    st.markdown('- Based on the similarity Score the model will suggest 10 movies')
    st.header(":red[Caution]")
    st.markdown('- Since Small set of Movies Around 4000 hollywood Movies Has been feeded to the System')
    st.markdown('- It cannot generate recommendation for some movies')
    st.markdown('- I Hope, in Future I will add more number of Movies so that the suggestion system can work better')
    
    st.success('Thank You!')
    
    
if selected == 'Recommendation':
    st.subheader(':red[Enter Your Favourite Hollywood Movie To Recommend]')
    col1,col2,col3 = st.columns(3)
    
    with col2:
        movie = st.text_input('', placeholder='Give a Movie')
    
    if movie:
        st.balloons()
        with col1:
            st.success('Suggested Movies')
            find_close_match = difflib.get_close_matches(movie, list_of_all_movies)
            
            close_match = find_close_match[0]
            # finding the index of movie with title
            index_of_movie = movies_data[movies_data.title == close_match]['index'].values[0]
            
            # Getting List of similar Movies
            # len(list(enumerate(similarity[index_of_movie])))
            similarity_score = list(enumerate(similarity[index_of_movie]))
            
            # Sorting the similarity score by index values
            similar_movies = sorted(similarity_score, key = lambda x: x[1], reverse =True)
            # printing the names of similar  by index values
            i=1
            for mv in similar_movies[1:]:
                index = mv[0]
                title = movies_data[movies_data.index== index]['title'].values[0]
                if i<=10:
                    st.write(i, '.', title)
                    i+=1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    