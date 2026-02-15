# Claire MCP Server

FastMCP server template with all patterns in one clean example.

## Setup

### 1. Get GitHub Token (Minimal Permissions)

**Option A: Fine-grained Token (Recommended)**
- Go to GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens
- Generate new token
- Select only YOUR specific repo(s)
- Repository permissions: **Contents: Read** (that's all!)
- Generate & copy token

**Option B: Classic Token**
- Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate new token with `repo` scope
- Copy the `ghp_xxxx...` token

### 2. Set Environment Variables
```bash
export GITHUB_TOKEN="ghp_yourtoken"
export GITHUB_REPO="yourusername/your-repo"
```

### 3. Configure Claude Desktop

**For Private Repos:**
```json
{
  "mcpServers": {
    "claire-tools": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://${GITHUB_TOKEN}@github.com/${GITHUB_REPO}.git",
        "claire-mcp"
      ]
    }
  }
}
```

**For Public Repos:**
```json
{
  "mcpServers": {
    "claire-tools": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yourusername/your-repo.git",
        "claire-mcp"
      ]
    }
  }
}
```

### 4. Restart Claude Desktop

## Development

The `process_file` example shows all FastMCP patterns:
- Required params (`path`)
- Optional with defaults (`output_format`, `max_lines`)
- Optional that can be None (`skip_patterns`)
- Boolean flags (`include_metadata`)
- Field descriptions for strict formats (`url`)
- Dict return type for structured data

Just replace with your own tools following the same patterns!

## Testing

```bash
# Clone and test locally
git clone https://github.com/yourusername/your-repo.git
cd your-repo

# Install FastMCP
pip install fastmcp

# Test the server
python server.py
```

## Troubleshooting

If you get import errors, test with uvx directly:
```bash
uvx --from git+https://github.com/yourusername/your-repo.git claire-mcp --help
```

This will show the actual Python error if there's an issue.
