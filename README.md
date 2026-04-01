# Youtube-to-webpage
Build a Generative AI system that converts YouTube transcripts into short summaries and then into structured article-style content (HTML/Markdown).
I’ve built a YouTube-to-Article Webpage Generator that transforms any YouTube video into a polished, Medium‑style article webpage in minutes! 🚀

How it works

Extract – Fetches transcript from a YouTube URL using YoutubeLoader.

Summarize – Uses Mistral AI (via LangChain) to condense the transcript into a professional article.

For long videos, it recursively summarizes in chunks, preserving technical details.

A LangChain agent with SummarizationMiddleware handles token limits seamlessly.

Render – Feeds the article into another LLM prompt that outputs complete HTML/CSS/JS – ready to deploy as a standalone webpage.

Tech Stack

Streamlit – Rapid UI development

LangChain – Orchestration, prompts, agents, and text splitting

Mistral AI (mistral‑small‑latest) – Core LLM for summarization and code generation

youtube‑transcript‑api – Transcript extraction

RecursiveCharacterTextSplitter – Chunking for long transcripts

Key Features
✅ Handles short & long transcripts automatically
✅ Generates responsive HTML/CSS/JS with dark/light toggle, smooth animations, and SEO tags
✅ Downloadable as individual files or ZIP archive
✅ Clean, first‑person professional tone, ignoring promotional content

This project was a deep dive into LangChain agents, recursive summarization, and prompt engineering for structured output. It’s a great example of how AI can repurpose video content into written formats with minimal manual effort.
