from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "vector_database"
BOOK_TITLE = "AI Engineering by Chip Huyen (O'Reilly, 2025)"
LLM_MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"

CURRICULUM = [
    {
        "id": 1,
        "title": "Introduction to AI Engineering",
        "description": "The landscape of AI engineering, how it differs from ML engineering, and the AI application stack.",
        "topics": [
            "What is AI Engineering?",
            "AI Engineer vs ML Engineer",
            "The AI Application Stack",
            "Foundation Models Overview",
            "AI Engineering Workflow",
        ],
    },
    {
        "id": 2,
        "title": "Foundation Models",
        "description": "Understanding how foundation models work, their capabilities, and limitations.",
        "topics": [
            "What are Foundation Models?",
            "LLMs vs Multimodal Models",
            "How Transformers Work",
            "Context Windows and Tokens",
            "Model Selection Criteria",
        ],
    },
    {
        "id": 3,
        "title": "Evaluation Methodology",
        "description": "How to rigorously evaluate AI systems for accuracy, safety, and quality.",
        "topics": [
            "Why Evaluation is Hard",
            "Automated vs Human Evaluation",
            "Evaluation Benchmarks",
            "LLM-as-Judge",
            "Building Evaluation Pipelines",
        ],
    },
    {
        "id": 4,
        "title": "Prompt Engineering",
        "description": "Techniques to effectively communicate with and get the best results from LLMs.",
        "topics": [
            "Zero-shot vs Few-shot Prompting",
            "Chain-of-Thought Prompting",
            "System Prompts & Role Design",
            "Prompt Injection & Security",
            "Advanced Prompting Strategies",
        ],
    },
    {
        "id": 5,
        "title": "RAG — Retrieval-Augmented Generation",
        "description": "Building systems that ground LLM responses in external knowledge.",
        "topics": [
            "Why RAG?",
            "Vector Databases & Embeddings",
            "Chunking Strategies",
            "Retrieval Techniques",
            "Advanced RAG Patterns",
        ],
    },
    {
        "id": 6,
        "title": "Agents & Tool Use",
        "description": "Building AI systems that can take actions, use tools, and reason autonomously.",
        "topics": [
            "What are AI Agents?",
            "Tool Use & Function Calling",
            "ReAct & Chain-of-Thought Agents",
            "Multi-Agent Systems",
            "Agent Safety & Reliability",
        ],
    },
    {
        "id": 7,
        "title": "Finetuning",
        "description": "Adapting foundation models to specific tasks and domains.",
        "topics": [
            "When to Finetune vs RAG",
            "Supervised Finetuning (SFT)",
            "RLHF & Preference Optimization",
            "Parameter-Efficient Finetuning (LoRA)",
            "Finetuning Best Practices",
        ],
    },
    {
        "id": 8,
        "title": "Dataset Engineering",
        "description": "Creating, curating, and annotating high-quality datasets for AI training.",
        "topics": [
            "Data Collection Strategies",
            "Data Annotation & Labeling",
            "Synthetic Data Generation",
            "Data Quality & Cleaning",
            "Open vs Closed Datasets",
        ],
    },
    {
        "id": 9,
        "title": "Inference & Serving",
        "description": "Optimizing AI models for production — latency, cost, and throughput.",
        "topics": [
            "Inference Bottlenecks",
            "Model Quantization",
            "KV Cache Optimization",
            "Batching Strategies",
            "Serving Infrastructure",
        ],
    },
    {
        "id": 10,
        "title": "Production AI Applications",
        "description": "Architecture patterns, observability, and lessons for building reliable AI products.",
        "topics": [
            "AI Application Architecture",
            "Observability & Monitoring",
            "Safety & Guardrails",
            "User Feedback Loops",
            "Lessons from Production",
        ],
    },
]
