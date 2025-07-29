import streamlit as st
import plotly.express as px
import altair as alt
import sys
import os
from dataset_inspection import *


st.set_page_config(
    page_title="Censorship Dataset Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Censorship Dataset Analysis Dashboard")
st.markdown("Analysis of LLM censorship patterns in generated Q&A dataset")

st.sidebar.header("Dataset Overview")

# Load data without caching
df = load_data()

summary_stats = get_summary_stats(df)
st.sidebar.metric("Total Questions", summary_stats['total_questions'])
st.sidebar.metric("Censored Questions", summary_stats['total_censored'])
st.sidebar.metric("Censorship Rate", f"{summary_stats['censorship_rate']:.1f}%")
st.sidebar.metric("Unique Subjects", summary_stats['unique_subjects'])

col1, col2 = st.columns([2, 1])

with col1:
    st.header("1. Censorship Category Distribution")
    
    label_dist = calculate_label_distribution(df)
    
    labels = list(label_dist.keys())
    counts = list(label_dist.values())
    
    fig_bar = px.bar(
        x=labels,
        y=counts,
        title="Distribution of Censorship Categories",
        labels={'x': 'Censorship Category', 'y': 'Count'},
        color=counts,
        color_continuous_scale='viridis'
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    st.header("2. Prompt Diversity")
    
    avg_similarity, similarities = calculate_prompt_diversity(df)
    
    st.metric(
        "Average Nearest Neighbor Similarity",
        f"{avg_similarity:.3f}",
        help="Lower values indicate more diverse prompts"
    )
    
    fig_hist = px.histogram(
        x=similarities,
        nbins=20,
        title="Prompt Similarity Distribution",
        labels={'x': 'Cosine Similarity', 'y': 'Count'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.header("3. Question Embeddings UMAP")
with st.spinner("Generating UMAP coordinates from question embeddings..."):
    # Use the function without caching
    df_umap = get_question_umap(df)

figure_question_umap = px.scatter(
    df_umap,
    x='question_umap_x',
    y='question_umap_y',
    hover_data=['question'],
    title="UMAP Visualization of Question Embeddings",
    labels={'question_umap_x': 'UMAP 1', 'question_umap_y': 'UMAP 2'}
)
figure_question_umap.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(figure_question_umap, use_container_width=True)


st.header("4. Response Embeddings UMAP")

with st.spinner("Generating UMAP coordinates from response embeddings..."):
    # Use the function without caching
    df_umap = get_response_umap(df)

figure_response_umap = px.scatter(
    df_umap,
    x='response_umap_x',
    y='response_umap_y',
    color='censorship_category',
    hover_data=['response'],
    title="UMAP Visualization of Response Embeddings",
    labels={'response_umap_x': 'UMAP 1', 'response_umap_y': 'UMAP 2'}
)
figure_response_umap.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(figure_response_umap, use_container_width=True)

st.header("5. Response Length Distribution")

col3, col4 = st.columns(2)

with col3:
    response_lengths = calculate_response_lengths(df)
    
    length_df = pd.DataFrame({'length': response_lengths})
    
    chart = alt.Chart(length_df).mark_bar().encode(
        alt.X('length:Q', bin=alt.Bin(maxbins=30), title='Response Length (words)'),
        alt.Y('count():Q', title='Count'),
        color=alt.value('steelblue')
    ).properties(
        title='Response Length Distribution',
        width=400,
        height=300
    )
    
    st.altair_chart(chart, use_container_width=True)

with col4:
    df_lengths = df.copy()
    df_lengths['response_length'] = response_lengths
    
    fig_box = px.box(
        df_lengths,
        x='censorship_category',
        y='response_length',
        title="Response Length by Category"
    )
    fig_box.update_xaxes(tickangle=45)
    st.plotly_chart(fig_box, use_container_width=True)

st.header("6. Distinctive N-grams by Category")

with st.spinner("Calculating distinctive n-grams..."):
    # Use the function without caching
    distinctive_ngrams = get_ngrams_data(df)

categories = list(distinctive_ngrams.keys())
tabs = st.tabs(categories)

for tab, category in zip(tabs, categories):
    with tab:
        ngrams_data = distinctive_ngrams[category]
        ngrams_df = pd.DataFrame(ngrams_data, columns=['N-gram', 'Log-odds Ratio'])
        ngrams_df['Rank'] = range(1, len(ngrams_df) + 1)
        
        st.dataframe(
            ngrams_df[['Rank', 'N-gram', 'Log-odds Ratio']],
            use_container_width=True,
            hide_index=True
        )

st.header("Data Export")

col5, col6 = st.columns(2)

with col5:
    if st.button("Download Dataset as CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="censorship_dataset.csv",
            mime="text/csv"
        )

with col6:
    if st.button("Refresh Data"):
        st.rerun()

st.markdown("---")
st.markdown("*Dashboard for analyzing LLM censorship patterns*")