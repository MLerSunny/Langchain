"""
Configuration schema definitions for validating YAML configuration files.
"""

from typing import Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum

class QuantizationType(str, Enum):
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"

class ComputeDType(str, Enum):
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"

class PaddingSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"

class SecurityConfig(BaseModel):
    """Schema for security configuration."""
    sanitize_input: bool = True
    sanitize_output: bool = True
    max_input_length: int = Field(ge=1)
    allowed_file_types: List[str]
    max_file_size: int = Field(ge=1)

class ModelOptimizationConfig(BaseModel):
    """Schema for model optimization configuration."""
    
    class QuantizationConfig(BaseModel):
        use_4bit: bool = False
        use_8bit: bool = False
        compute_dtype: ComputeDType = ComputeDType.FLOAT16
        use_double_quant: bool = True
        quant_type: QuantizationType = QuantizationType.NF4

    class LoRAConfig(BaseModel):
        enabled: bool = True
        rank: int = Field(ge=1, le=256)
        alpha: int = Field(ge=1, le=256)
        dropout: float = Field(ge=0.0, le=1.0)
        target_modules: List[str]
        bias: str = "none"
        task_type: str = "CAUSAL_LM"

    class MemoryConfig(BaseModel):
        max_memory: str
        offload_folder: str
        use_cache: bool = False
        gradient_checkpointing: bool = True
        pin_memory: bool = False
        dataloader_num_workers: int = Field(ge=0)

    class TrainingConfig(BaseModel):
        fp16: bool = True
        optim: str
        report_to: str = "none"
        remove_unused_columns: bool = False
        gradient_checkpointing: bool = True
        dataloader_num_workers: int = Field(ge=0)
        dataloader_pin_memory: bool = False

    class ModelLoadingConfig(BaseModel):
        torch_dtype: ComputeDType = ComputeDType.FLOAT16
        device_map: str = "auto"
        use_cache: bool = False
        max_memory: str
        offload_folder: str

    class TokenizerConfig(BaseModel):
        pad_token: str
        model_max_length: int = Field(ge=1)
        padding_side: PaddingSide = PaddingSide.RIGHT
        truncation_side: PaddingSide = PaddingSide.RIGHT

    quantization: QuantizationConfig
    lora: LoRAConfig
    memory: MemoryConfig
    training: TrainingConfig
    model_loading: ModelLoadingConfig
    tokenizer: TokenizerConfig

    @validator('memory.max_memory', 'model_loading.max_memory', check_fields=False)
    def validate_memory_format(cls, v: str) -> str:
        """Validate memory format (e.g., '8GiB')."""
        if not v.endswith(('B', 'KiB', 'MiB', 'GiB', 'TiB')):
            raise ValueError("Memory must end with B, KiB, MiB, GiB, or TiB")
        return v

class RAGConfig(BaseModel):
    """Schema for RAG configuration."""
    
    class ChunkingConfig(BaseModel):
        chunk_size: int = Field(ge=100, le=2000)
        chunk_overlap: int = Field(ge=0, le=500)
        max_chunks: int = Field(ge=1)
        chunk_strategy: str
        min_chunk_size: int = Field(ge=1)
        max_chunk_size: int = Field(ge=1)
        split_by: str

    class RetrievalConfig(BaseModel):
        similarity_threshold: float = Field(ge=0.0, le=1.0)
        max_results: int = Field(ge=1)
        rerank_results: bool
        rerank_model: str
        rerank_top_k: int = Field(ge=1)
        diversity_penalty: float = Field(ge=0.0, le=1.0)
        max_marginal_relevance: bool
        mmr_lambda: float = Field(ge=0.0, le=1.0)

    class VectorStoreConfig(BaseModel):
        persist_directory: str
        collection_name: str
        dimension: int = Field(ge=1)
        index_params: Dict[str, Any]

    class EmbeddingsConfig(BaseModel):
        model: str
        cache_dir: str
        batch_size: int = Field(ge=1)
        normalize: bool
        truncate: str
        max_length: int = Field(ge=1)

    class CacheConfig(BaseModel):
        enabled: bool
        ttl: int = Field(ge=0)
        max_size: int = Field(ge=1)
        strategy: str
        persist: bool
        persist_dir: str

    class QueryConfig(BaseModel):
        preprocess: bool
        expand_queries: bool
        max_query_length: int = Field(ge=1)
        query_template: str
        system_prompt: str

    class GenerationConfig(BaseModel):
        model: str
        temperature: float = Field(ge=0.0, le=2.0)
        max_tokens: int = Field(ge=1)
        top_p: float = Field(ge=0.0, le=1.0)
        frequency_penalty: float = Field(ge=-2.0, le=2.0)
        presence_penalty: float = Field(ge=-2.0, le=2.0)
        stop_sequences: List[str]
        include_sources: bool
        max_source_tokens: int = Field(ge=1)

    class MonitoringConfig(BaseModel):
        enabled: bool
        log_level: str
        metrics: List[str]
        alerts: Dict[str, Any]

    class ErrorHandlingConfig(BaseModel):
        retry_attempts: int = Field(ge=0)
        retry_delay: int = Field(ge=0)
        fallback_responses: Dict[str, str]

    class OptimizationConfig(BaseModel):
        parallel_processing: bool
        max_workers: int = Field(ge=1)
        batch_processing: bool
        batch_size: int = Field(ge=1)
        prefetch_factor: int = Field(ge=1)
        pin_memory: bool

    class FeaturesConfig(BaseModel):
        semantic_search: bool
        hybrid_search: bool
        query_expansion: bool
        result_reranking: bool
        source_citations: bool
        answer_validation: bool
        context_window: bool

    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    vector_store: VectorStoreConfig
    embeddings: EmbeddingsConfig
    cache: CacheConfig
    query: QueryConfig
    generation: GenerationConfig
    monitoring: MonitoringConfig
    error_handling: ErrorHandlingConfig
    security: SecurityConfig
    optimization: OptimizationConfig
    features: FeaturesConfig 