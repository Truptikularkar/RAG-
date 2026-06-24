import re
from typing import List, Dict, Any

class Chunker:
    @staticmethod
    def fixed_size_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Splits text into fixed size chunks with overlap, tracking original string indices."""
        chunks = []
        if not text:
            return chunks
            
        if chunk_size <= 0:
            chunk_size = 500
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2

        step = chunk_size - chunk_overlap
        text_len = len(text)
        
        i = 0
        chunk_idx = 0
        while i < text_len:
            end = min(i + chunk_size, text_len)
            chunk_text = text[i:end]
            
            # Avoid empty or whitespace-only chunks
            if chunk_text.strip():
                chunks.append({
                    "id": chunk_idx,
                    "text": chunk_text,
                    "start_idx": i,
                    "end_idx": end
                })
                chunk_idx += 1
                
            if end == text_len:
                break
            i += step
            
        return chunks

    @staticmethod
    def recursive_character_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
        """Splits text recursively using separators: paragraphs, sentences, words, characters.
        
        Preserves original document index positions for each chunk.
        """
        # Separators in order of preference (excluding empty string to avoid ValueError)
        separators = ["\n\n", "\n", ". ", " "]
        
        chunks = []
        if not text:
            return chunks

        if chunk_size <= 0:
            chunk_size = 500
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size // 2

        # A helper structure to store splits with their absolute original start index
        class TextSplit:
            def __init__(self, content: str, start: int):
                self.content = content
                self.start = start
                self.end = start + len(content)

        # Initial queue with the full text
        queue = [TextSplit(text, 0)]
        final_splits = []

        while queue:
            current = queue.pop(0)
            
            # If the current split fits inside chunk size, keep it
            if len(current.content) <= chunk_size:
                if current.content.strip():
                    final_splits.append(current)
                continue
                
            # Find the best separator that works
            found_separator = False
            for sep in separators:
                if sep in current.content:
                    # Split by separator
                    parts = current.content.split(sep)
                    
                    # Reconstruct splits with correct offsets
                    current_offset = current.start
                    temp_splits = []
                    
                    for idx, part in enumerate(parts):
                        # Add separator back (except for the last part)
                        part_content = part + (sep if idx < len(parts) - 1 else "")
                        if part_content:
                            temp_splits.append(TextSplit(part_content, current_offset))
                        current_offset += len(part_content)
                        
                    # If this separator didn't split the content into multiple parts, try next separator
                    if len(temp_splits) <= 1:
                        continue
                        
                    # Queue the sub-splits to be checked/further split
                    # Insert at the beginning of the queue to maintain ordering (DFS-like)
                    queue = temp_splits + queue
                    found_separator = True
                    break
            
            # If no separator splits the block, we resort to hard character splitting
            if not found_separator:
                # Force chunking by character size
                i = 0
                step = chunk_size - chunk_overlap
                if step <= 0:
                    step = chunk_size // 2
                while i < len(current.content):
                    end = min(i + chunk_size, len(current.content))
                    part_content = current.content[i:end]
                    final_splits.append(TextSplit(part_content, current.start + i))
                    if end == len(current.content):
                        break
                    i += step

        # Merge splits to fill up to chunk_size as much as possible, respecting overlap
        merged_chunks = []
        if not final_splits:
            return []

        # Simple merging algorithm that accumulates text splits
        current_accum = []
        current_len = 0
        
        for split in final_splits:
            if current_len + len(split.content) <= chunk_size:
                current_accum.append(split)
                current_len += len(split.content)
            else:
                # Flush the current accumulator
                if current_accum:
                    merged_chunks.append(Chunker._create_merged_chunk(current_accum))
                
                # Keep splits that overlap with next chunk
                # Find splits at the tail of current_accum that fit in the overlap window
                overlap_accum = []
                overlap_len = 0
                for s in reversed(current_accum):
                    if overlap_len + len(s.content) <= chunk_overlap:
                        overlap_accum.insert(0, s)
                        overlap_len += len(s.content)
                    else:
                        break
                
                current_accum = overlap_accum + [split]
                current_len = sum(len(s.content) for s in current_accum)
                
        if current_accum:
            merged_chunks.append(Chunker._create_merged_chunk(current_accum))

        # Build clean output format
        output = []
        for idx, mc in enumerate(merged_chunks):
            output.append({
                "id": idx,
                "text": mc["text"],
                "start_idx": mc["start_idx"],
                "end_idx": mc["end_idx"]
            })
            
        return output

    @staticmethod
    def _create_merged_chunk(splits: List[Any]) -> Dict[str, Any]:
        """Combines consecutive TextSplit objects into a single chunk dictionary."""
        text = "".join(s.content for s in splits)
        start_idx = splits[0].start
        end_idx = splits[-1].end
        return {
            "text": text,
            "start_idx": start_idx,
            "end_idx": end_idx
        }
