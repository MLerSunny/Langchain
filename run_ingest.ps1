# Set environment variables for the ingestion process
$env:PYTHONMALLOC = "malloc"
$env:CYGWIN = "heap_chunk_in_mb=2048,tp_num_c_bufs=1024"
$env:LOG_LEVEL = "DEBUG"

# Run the bulk ingestion script with the provided arguments
python ingest/bulk_ingest.py --source-dir data/raw/auto --batch-size 5 --mode append
python ingest/bulk_ingest.py --source-dir data/training --batch-size 5 --mode append