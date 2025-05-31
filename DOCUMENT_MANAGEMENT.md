# Document Management Guide

This guide covers how to use the enhanced document management features to add, view, and manage documents in your RAG system.

## Command Line Batch Processing

For processing multiple directories or large document sets, the command-line interface is recommended:

### Single Directory Processing

```bash
python scripts/ingest.py --input-dir "data/raw/your_documents"
```

### Batch Processing Multiple Directories

The new batch processing script can process multiple directories at once:

```bash
python scripts/batch_ingest.py --input-dirs "data/raw/folder1" "data/raw/folder2" "data/raw/folder3"
```

### Advanced Options

The batch processing script supports several options:

```bash
python scripts/batch_ingest.py \
  --input-dirs "data/raw/folder1" "data/raw/folder2" \
  --recursive \
  --chunk-size 512 \
  --chunk-overlap 50 \
  --max-chunks 200 \
  --embedding-model "all-MiniLM-L6-v2"
```

Options explanation:
- `--input-dirs`: One or more directories containing documents to process
- `--recursive`: Process subdirectories recursively
- `--chunk-size`: Size of document chunks in tokens (default: 1024)
- `--chunk-overlap`: Overlap between chunks in tokens (default: 128)
- `--max-chunks`: Maximum number of chunks to process per directory
- `--embedding-model`: Model to use for creating embeddings

## Document Management Tab

The Document Management tab provides a user-friendly interface for managing your vector database:

### View Documents

The "View Documents" tab lets you:
- See all documents in your vector database
- Filter documents by source or topic
- View document distribution by source
- Delete individual documents

### Add Documents

The "Add Documents" tab lets you:
- Upload files directly through the UI
- Process documents from an existing directory
- Set custom chunking parameters
- Add metadata like topics and categories

### Search Documents

The "Search Documents" tab lets you:
- Search documents semantically
- View the most relevant documents for your query
- See document metadata and relevance scores

## Tips for Effective Document Management

1. **Organize your documents**: Keep related documents in the same directory for easier batch processing
2. **Choose appropriate chunk sizes**: 
   - Smaller chunks (256-512 tokens) work better for specific facts
   - Larger chunks (1024-2048 tokens) preserve more context
3. **Add meaningful metadata**: 
   - Set topics and categories to make filtering easier
   - Consistent naming helps with organization
4. **Regular maintenance**:
   - Remove outdated documents
   - Update documents when information changes
5. **Monitor database size**:
   - Large vector databases can impact performance
   - Consider using separate collections for different domains

## Troubleshooting

If you encounter issues:

1. **Document not appearing in search results**:
   - Verify it was properly added to the vector database
   - Check if the document is properly formatted

2. **Slow performance**:
   - Reduce the number of chunks by using a larger chunk size
   - Use more specific metadata filters when searching

3. **Memory errors during processing**:
   - Reduce the `max_chunks` parameter
   - Process documents in smaller batches

4. **Document content not properly extracted**:
   - Check if the document format is supported
   - Try converting to a different format (e.g., PDF to TXT) 