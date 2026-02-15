from fastmcp import FastMCP
from typing import Dict, Any, List, Optional, Annotated
from pydantic import Field

mcp = FastMCP("claire-mcp")

@mcp.tool
def process_file(
    path: str,
    output_format: str = "json",
    max_lines: int = 1000,
    skip_patterns: Optional[List[str]] = None,
    include_metadata: bool = False,
    url: Annotated[str, Field(description="Full direct URL to file, used over path if provided")] = None
) -> Dict[str, Any]:
    """Process local or remote file with various output options"""
    return {
        "file": path,
        "lines_processed": 850,
        "format": output_format,
        "skipped": len(skip_patterns) if skip_patterns else 0
    }

def main():
    """Main entry point for the MCP server."""
    # Run the FastMCP server
    mcp.run()


if __name__ == "__main__":
    main()
