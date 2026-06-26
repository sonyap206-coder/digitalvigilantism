import argparse
import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def clean_comments(df, text_col="text"):
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    return df

def save_wordclouds(topic_model, output_prefix):
    pdf_path = f"{output_prefix}_wordclouds.pdf"

    with PdfPages(pdf_path) as pdf:

        for topic_id in topic_model.get_topic_info()["Topic"]:

            if topic_id == -1:
                continue

            words = dict(topic_model.get_topic(topic_id))

            wc = WordCloud(
                width=800,
                height=400,
                background_color="white"
            )

            wc.generate_from_frequencies(words)

            fig, ax = plt.subplots(figsize=(11, 8.5))

            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")

            # Add topic title
            ax.set_title(
                f"Topic {topic_id}",
                fontsize=18,
                pad=20
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved word clouds to {pdf_path}")

def save_topic_size_chart(topic_info, output_prefix):
    topic_info_filtered = topic_info[topic_info["Topic"] != -1]
    plt.figure(figsize=(10, 5))
    plt.bar(
        topic_info_filtered["Topic"].astype(str),
        topic_info_filtered["Count"]
    )
    plt.xlabel("Topic")
    plt.ylabel("Number of Comments")
    plt.title("Comment Count per Topic")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_topic_sizes.png", dpi=150)
    plt.close()
    print("Saved topic size chart.")

def save_intertopic_map(topic_model, output_prefix):
    fig = topic_model.visualize_topics()
    fig.write_html(f"{output_prefix}_intertopic_map.html")
    print("Saved intertopic distance map.")

def save_heatmap(topic_model, output_prefix):
    fig = topic_model.visualize_heatmap()
    fig.write_html(f"{output_prefix}_heatmap.html")
    print("Saved similarity heatmap.")

def save_document_map(topic_model, comments, embeddings, output_prefix):
    fig = topic_model.visualize_documents(comments, embeddings=embeddings)
    fig.write_html(f"{output_prefix}_document_map.html")
    print("Saved document map.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input CSV file")
    parser.add_argument("-o", "--output", default="bertopic_results", help="Output prefix")
    parser.add_argument("--text-col", default="text", help="Column name for comments")
    args = parser.parse_args()

    # Load and clean
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df = clean_comments(df, text_col=args.text_col)
    comments = df[args.text_col].tolist()
    print(f"Loaded {len(comments)} comments.")

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Extract embeddings explicitly so we can reuse them for visualize_documents
    # and save them to disk for recovery runs (generate_heatmap.py)
    print("Generating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(comments, show_progress_bar=True)
    # np.save(f"{args.output}_embeddings.npy", embeddings)
    # print(f"Saved embeddings to {args.output}_embeddings.npy")

    # Build and fit BERTopic
    # nr_topics=None means fully automatic
    # nr_topics=10 forces exactly 10 topics
    topic_model = BERTopic(
        language="english",
        ctfidf_model=ctfidf_model,
        min_topic_size= 50,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(comments, embeddings)
    # If there are lots of outliers:
    new_topics = topic_model.reduce_outliers(
        comments,
        topics
    )

    topic_model.update_topics(
        comments,
        topics=new_topics
    )

    # Reduce to approximately 15 topics by merging similar ones
    topic_model.reduce_topics(comments, nr_topics=15)
    topics = topic_model.topics_  # update topic assignments after reduction

    # Save comment-level results
    df["topic"] = topics
    df["topic_probability"] = probs
    df.to_csv(f"{args.output}_comments.csv", index=False, encoding="utf-8-sig")
    print(f"Saved comment assignments to {args.output}_comments.csv")

    # Save topic info (keywords per topic)
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(f"{args.output}_topic_info.csv", index=False, encoding="utf-8-sig")
    print(f"Saved topic info to {args.output}_topic_info.csv")

    # Print top keywords per topic to terminal
    print("\nTop keywords per topic:")
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            continue  # -1 is BERTopic's "outlier" bucket
        words = [word for word, _ in topic_model.get_topic(topic_id)]
        print(f"  Topic {topic_id}: {', '.join(words)}")

    # Save model so it can be reloaded later without re-running
    # topic_model.save(f"{args.output}_model", serialization="pickle")
    # print(f"Saved model to {args.output}_model")

    # Visualizations
    save_wordclouds(topic_model, args.output)
    save_topic_size_chart(topic_info, args.output)
    save_intertopic_map(topic_model, args.output)
    save_heatmap(topic_model, args.output)
    save_document_map(topic_model, comments, embeddings, args.output)

if __name__ == "__main__":
    main()