"""
Simplified dataset generation pipeline.
Processes each article ONCE to generate maximum redundancy upfront.
"""

import argparse
import random
import time
from pathlib import Path
from typing import List
from tqdm import tqdm

from core.wikipedia.fetcher import WikipediaFetcher, WikiArticle
from core.wikipedia.chunk_creator import ChunkCreator, Chunk
from core.embedder import EmbeddingGenerator, EmbeddedChunk
from core.vector_store import VectorStore
from core.checkpoint_manager import CheckpointManager


def load_article_list(file_path: str) -> List[str]:
    """Load list of Wikipedia article titles."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                articles.append(line)
    return articles


def fetch_articles(
    article_titles: List[str],
    num_articles: int,
    fetcher: WikipediaFetcher
) -> tuple[List[WikiArticle], List[WikiArticle]]:
    """
    Fetch Wikipedia articles for prompts and noise.

    Args:
        article_titles: List of article titles to fetch
        num_articles: Number of prompt articles needed
        fetcher: WikipediaFetcher instance

    Returns:
        (prompt_articles, noise_articles)
    """
    successful_articles = []
    failed_count = 0

    # Need 2x articles: half for prompts, half for noise
    target_count = num_articles * 2

    for title in tqdm(article_titles, desc="Fetching articles"):
        if len(successful_articles) >= target_count:
            break

        article = fetcher.fetch_article(title)

        if article:
            validation = fetcher.validate_article_quality(article)
            if validation['valid']:
                successful_articles.append(article)
            else:
                failed_count += 1
        else:
            failed_count += 1

    # Split: first half for prompts, second half for noise
    prompt_articles = successful_articles[:num_articles]
    noise_articles = successful_articles[num_articles:target_count]

    print(f"Stage 1 complete: {len(successful_articles)} articles fetched, {failed_count} failed")
    print(f"  Prompt articles: {len(prompt_articles)}")
    print(f"  Noise articles: {len(noise_articles)}")

    return prompt_articles, noise_articles


def create_chunks(
    prompt_articles: List[WikiArticle],
    noise_articles: List[WikiArticle],
    max_redundancy: int,
    chunk_creator: ChunkCreator
) -> List[Chunk]:
    """
    Create all chunks (gold + redundant + noise + prompts).

    Args:
        prompt_articles: Articles to create prompts from
        noise_articles: Articles to build noise pool from
        max_redundancy: Maximum redundancy level (creates max_redundancy + 1 chunks per aspect)
        chunk_creator: ChunkCreator instance

    Returns:
        List of all Chunk objects
    """
    # Build noise pool
    noise_pool_size = chunk_creator.build_noise_pool(noise_articles)
    print(f"Built noise pool: {noise_pool_size} paragraphs")

    all_chunks = []
    gold_count = 0
    noise_count = 0
    prompt_count = 0
    failed_count = 0

    # Process each prompt article
    for article in tqdm(prompt_articles, desc="Creating chunks"):
        result = chunk_creator.create_chunks_for_article(article)

        if result is None:
            failed_count += 1
            continue

        prompt_chunk, gold_chunks, noise_chunks = result

        # Add to collection
        all_chunks.append(prompt_chunk)
        all_chunks.extend(gold_chunks)
        all_chunks.extend(noise_chunks)

        # Update counts
        prompt_count += 1
        gold_count += len(gold_chunks)
        noise_count += len(noise_chunks)

    print(f"Stage 2 complete: {len(all_chunks)} chunks created")
    print(f"  Prompts: {prompt_count}")
    print(f"  Gold chunks: {gold_count} (base + redundant)")
    print(f"  Noise chunks: {noise_count}")
    print(f"  Failed articles: {failed_count}")

    return all_chunks


def embed_chunks(
    chunks: List[Chunk],
    embedding_generator: EmbeddingGenerator
) -> List[EmbeddedChunk]:
    """
    Generate embeddings for all chunks.

    Args:
        chunks: List of Chunk objects
        embedding_generator: EmbeddingGenerator instance

    Returns:
        List of EmbeddedChunk objects
    """
    embedded_chunks = embedding_generator.embed_chunks(chunks)

    print(f"Stage 3 complete: {len(embedded_chunks)} embeddings generated")

    return embedded_chunks


def upload_to_vector_store(
    embedded_chunks: List[EmbeddedChunk],
    vector_store: VectorStore,
    checkpoint_manager: CheckpointManager
):
    """
    Upload chunks with embeddings to ChromaDB.

    Args:
        embedded_chunks: List of EmbeddedChunk objects
        vector_store: VectorStore instance
        checkpoint_manager: CheckpointManager instance
    """
    vector_store.add(embedded_chunks)

    # Mark upload as complete
    checkpoint_manager.mark_upload_complete()

    print(f"Stage 4 complete: {len(embedded_chunks)} chunks uploaded")

    # Show collection stats
    stats = vector_store.get_statistics()
    print("\nChroma Collection Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Wikipedia aspect dataset with redundancy"
    )

    parser.add_argument(
        '--num-articles',
        type=int,
        default=100,
        help='Number of prompt articles to process (default: 100)'
    )

    parser.add_argument(
        '--max-redundancy',
        type=int,
        default=5,
        help='Maximum redundancy level (default: 5, creates 6 chunks per aspect)'
    )

    parser.add_argument(
        '--article-list',
        type=str,
        default='./data/wikipedia/wiki_articles.txt',
        help='Path to article list file'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: process only 5 articles'
    )

    parser.add_argument(
        '--clear-checkpoints',
        action='store_true',
        help='Clear all checkpoints and start fresh'
    )

    parser.add_argument(
        '--clear-db',
        action='store_true',
        help='Clear ChromaDB collection before uploading'
    )

    args = parser.parse_args()

    # Configuration
    num_articles = 5 if args.test else args.num_articles
    max_redundancy = args.max_redundancy

    print("=" * 70)
    print("Wikipedia Aspect Dataset Generation - Simplified Pipeline")
    print("=" * 70)
    print(f"Articles: {num_articles}")
    print(f"Max redundancy: {max_redundancy} (creates {max_redundancy + 1} chunks per aspect)")
    print("=" * 70)

    # Initialize components
    checkpoint_manager = CheckpointManager(checkpoint_dir='./data/wikipedia/checkpoints')
    fetcher = WikipediaFetcher(cache_dir='./data/wikipedia/cache')
    chunk_creator = ChunkCreator(max_redundancy=max_redundancy)
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(collection_name='wiki_aspects', persist_directory='./data/wikipedia/db')

    # Clear checkpoints if requested
    if args.clear_checkpoints:
        checkpoint_manager.clear()
        print("Checkpoints cleared")

    # Clear DB if requested
    if args.clear_db:
        vector_store.clear()
        print("ChromaDB collection cleared")

    # Track total time
    start_time = time.time()

    # Stage 1: Fetch articles
    print("\n" + "=" * 70)
    print("STAGE 1: Fetching Wikipedia Articles")
    print("=" * 70)

    if not checkpoint_manager.has_stage('articles'):
        article_titles = load_article_list(args.article_list)
        print(f"Loaded {len(article_titles)} article titles")

        # Shuffle for randomness
        random.seed(42)
        random.shuffle(article_titles)

        prompt_articles, noise_articles = fetch_articles(
            article_titles,
            num_articles,
            fetcher
        )

        # Save to checkpoint
        checkpoint_manager.save_stage('articles', prompt_articles + noise_articles)
        checkpoint_manager.status['num_prompt_articles'] = len(prompt_articles)
    else:
        print("Loading articles from checkpoint...")
        all_articles = checkpoint_manager.load_stage('articles')
        num_prompt = checkpoint_manager.status.get('num_prompt_articles', num_articles)
        prompt_articles = all_articles[:num_prompt]
        noise_articles = all_articles[num_prompt:]
        print(f"Loaded {len(prompt_articles)} prompt articles, {len(noise_articles)} noise articles")

    # Stage 2: Create chunks
    print("\n" + "=" * 70)
    print("STAGE 2: Creating Chunks with Redundancy")
    print("=" * 70)

    if not checkpoint_manager.has_stage('chunks'):
        chunks = create_chunks(
            prompt_articles,
            noise_articles,
            max_redundancy,
            chunk_creator
        )

        # Save to checkpoint
        checkpoint_manager.save_stage('chunks', chunks)
    else:
        print("Loading chunks from checkpoint...")
        chunks = checkpoint_manager.load_stage('chunks')
        print(f"Loaded {len(chunks)} chunks")

    # Stage 3: Generate embeddings
    print("\n" + "=" * 70)
    print("STAGE 3: Generating Embeddings")
    print("=" * 70)

    if not checkpoint_manager.has_stage('embeddings'):
        embedded_chunks = embed_chunks(chunks, embedding_generator)

        # Save to checkpoint
        checkpoint_manager.save_stage('embeddings', embedded_chunks)
    else:
        print("Loading embeddings from checkpoint...")
        embedded_chunks = checkpoint_manager.load_stage('embeddings')
        print(f"Loaded {len(embedded_chunks)} embeddings")

    # Stage 4: Upload to Chroma
    print("\n" + "=" * 70)
    print("STAGE 4: Uploading to ChromaDB")
    print("=" * 70)

    upload_to_vector_store(embedded_chunks, vector_store, checkpoint_manager)

    # Final summary
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n" + "=" * 70)
    print("Dataset Generation Complete!")
    print("=" * 70)
    print(f"Total time: {minutes}m {seconds}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
