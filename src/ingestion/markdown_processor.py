"""Markdown processor to separate text and table chunks."""
import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TextSegment:
    """A text segment from markdown content."""
    content: str
    start_position: int
    end_position: int


@dataclass 
class TableChunk:
    """A table chunk from markdown content."""
    content: str
    start_position: int
    end_position: int
    row_count: int
    column_count: int


class MarkdownProcessor:
    """
    Process markdown content to separate text and table chunks.
    
    Tables are extracted as separate chunks to allow LLM to reason
    on structured data more effectively.
    """
    
    # Regex pattern for markdown tables
    # Matches: | header | header |
    #          |--------|--------|
    #          | data   | data   |
    TABLE_PATTERN = re.compile(
        r'(\|[^\n]+\|\n)'           # Header row
        r'(\|[-:\s|]+\|\n)'         # Separator row  
        r'((?:\|[^\n]+\|\n?)+)',    # Data rows
        re.MULTILINE
    )
    
    # Pattern for HTML tables (when output_tables_as_html=True)
    HTML_TABLE_PATTERN = re.compile(
        r'<table[^>]*>.*?</table>',
        re.DOTALL | re.IGNORECASE
    )
    
    def extract_tables(self, markdown: str) -> List[TableChunk]:
        """
        Extract all markdown tables from content.
        
        Args:
            markdown: Markdown content
            
        Returns:
            List of TableChunk objects
        """
        tables = []
        
        # Find markdown tables
        for match in self.TABLE_PATTERN.finditer(markdown):
            table_content = match.group(0)
            rows = table_content.strip().split('\n')
            
            # Count rows (excluding separator) and columns
            row_count = len([r for r in rows if not re.match(r'^\|[-:\s|]+\|$', r)])
            col_count = len(rows[0].split('|')) - 2 if rows else 0  # -2 for empty strings at start/end
            
            tables.append(TableChunk(
                content=table_content.strip(),
                start_position=match.start(),
                end_position=match.end(),
                row_count=row_count,
                column_count=col_count
            ))
        
        # Find HTML tables
        for match in self.HTML_TABLE_PATTERN.finditer(markdown):
            table_content = match.group(0)
            # Estimate row count from <tr> tags
            row_count = table_content.lower().count('<tr')
            col_count = table_content.lower().count('<th') or table_content.lower().count('<td')
            
            tables.append(TableChunk(
                content=table_content,
                start_position=match.start(),
                end_position=match.end(),
                row_count=row_count,
                column_count=col_count
            ))
        
        # Sort by position
        tables.sort(key=lambda t: t.start_position)
        
        logger.debug(f"Extracted {len(tables)} tables from markdown")
        return tables
    
    def extract_text_segments(self, markdown: str, tables: List[TableChunk]) -> List[TextSegment]:
        """
        Extract text content excluding tables.
        
        Args:
            markdown: Original markdown content
            tables: List of extracted tables
            
        Returns:
            List of TextSegment objects
        """
        if not tables:
            return [TextSegment(
                content=markdown.strip(),
                start_position=0,
                end_position=len(markdown)
            )]
        
        segments = []
        current_pos = 0
        
        for table in tables:
            # Get text before this table
            if table.start_position > current_pos:
                text_content = markdown[current_pos:table.start_position].strip()
                if text_content:
                    segments.append(TextSegment(
                        content=text_content,
                        start_position=current_pos,
                        end_position=table.start_position
                    ))
            current_pos = table.end_position
        
        # Get remaining text after last table
        if current_pos < len(markdown):
            text_content = markdown[current_pos:].strip()
            if text_content:
                segments.append(TextSegment(
                    content=text_content,
                    start_position=current_pos,
                    end_position=len(markdown)
                ))
        
        logger.debug(f"Extracted {len(segments)} text segments from markdown")
        return segments
    
    def process(self, markdown: str) -> Tuple[List[TextSegment], List[TableChunk]]:
        """
        Process markdown content to separate text and tables.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Tuple of (text_segments, table_chunks)
        """
        tables = self.extract_tables(markdown)
        text_segments = self.extract_text_segments(markdown, tables)
        
        logger.info(
            f"Processed markdown: {len(text_segments)} text segments, "
            f"{len(tables)} tables"
        )
        
        return text_segments, tables


# Singleton instance
_markdown_processor: MarkdownProcessor = None


def get_markdown_processor() -> MarkdownProcessor:
    """Get or create the MarkdownProcessor singleton."""
    global _markdown_processor
    if _markdown_processor is None:
        _markdown_processor = MarkdownProcessor()
    return _markdown_processor
