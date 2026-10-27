<p align="center">
  <img src="logo/Ghana%20Search%20logo@1x.png" width="340" alt="Ghana Search Logo">
</p>

# Ghana Search: Ghana's Information Retrieval Engine

[![M8ven Score](https://m8ven.ai/badge/mcp/ghanaopenai-ghana-search-10yfiv?v=cb894bdd32b11a61c49f983a1e312e29)](https://m8ven.ai/mcp/ghanaopenai-ghana-search-10yfiv?s=readme)

Ghana Search is an information retrieval engine and Model Context Protocol (MCP) server that allows users, developers, and AI agents to search verified Ghanaian digital sources in real time.

Live search portal: [https://search.ghanaopenai.org](https://search.ghanaopenai.org)

---

## Supported Sources

Ghana Search queries the public search APIs of over 430 Ghanaian institutions and repositories, returning direct content URLs and downloadable PDFs rather than generic homepages:

- **Government Portals (164 websites)**: Direct search across ministries, departments, and agencies including the Ghana Revenue Authority (GRA), Food and Drugs Authority (FDA), Ghana Standards Authority (GSA), Ministry of Lands and Natural Resources, Lands Commission, and local assemblies.
- **District Assemblies (GhanaDistricts)**: News, development projects, and assembly reports across all 261 Ghanaian district assemblies.
- **Court Judgments and Legislation (GhaLII)**: Supreme Court, Court of Appeal, High Court rulings, Acts of Parliament, and Decrees with direct document and source download links.
- **Parliament of Ghana**: Official parliamentary Bills, Hansard records, and committee reports with direct PDF downloads.
- **University Theses and Research**: Academic research abstracts and direct PDF thesis download links from KNUST, University of Ghana (UGSpace), and University of Cape Coast (UCC).
- **National News Outlets**: Real-time articles from B&FT Online, 3News, Citi Newsroom, Ghana News Agency (GNA), Yen.com.gh, The Chronicle, Ghana Business News, and Starr FM.
- **Ghana Yellow Pages & Commercial Directory (GhanaBusinessWeb)**: Direct search across verified Ghanaian commercial enterprises, pharmacies, hospitals, legal chambers, solar energy suppliers, hotels, and contractors with physical addresses, telephone contacts, and directory listings.
- **Ghanaian Reference**: Scoped reference and historical background from Wikipedia (Ghana).

---

## Web Search Interface

Visit [https://search.ghanaopenai.org](https://search.ghanaopenai.org) to search directly from your browser with three specialized modes:

1. **National Knowledge (Core)**:
   - Queries 164 verified government portals, university thesis repositories (KNUST/UG/UCC), Supreme Court rulings (GhaLII), parliamentary records, and national news networks.
   - Title-first & recent-first ranking with direct PDF download links.

2. **Business Directory (Yellow Pages)**:
   - **Two-way lookup**: Search by product/commodity to find suppliers and stores (e.g. *"solar panels"*, *"tractor parts"*, *"pharmacies Kumasi"*), OR search by business name (e.g. *"Ernest Chemists"*, *"Solar 4 Ghana"*) to get branches, physical addresses, and direct phone call links.

3. **Price Finder (Marketplace Intelligence)**:
   - Search active Ghanaian classifieds ads and calculate real-time pricing metrics in Ghana Cedis (`GH₵`): average market price, price range (min/max), and active product listings.

---

## Querying the APIs

Ghana Search offers dedicated endpoints for each capability:

### 1. National Knowledge Retrieval (`POST /api/search`)

```bash
curl -X POST https://search.ghanaopenai.org/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ghana Revenue Authority VAT and electronic invoicing requirements",
    "top_k": 5
  }'
```

### 2. Business Directory & Supplier Locator (`POST /api/business/search`)

```bash
curl -X POST https://search.ghanaopenai.org/api/business/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "tractor parts Accra",
    "top_k": 5
  }'
```

### 3. Classifieds & Market Price Finder (`POST /api/prices/search`)

```bash
curl -X POST https://search.ghanaopenai.org/api/prices/search \
  -H "Content-Type: application/json" \
  -d '{
    "product": "MacBook Pro",
    "limit": 5
  }'
```

```json
{
  "status": "success",
  "data": {
    "query": "MacBook Pro",
    "sample_size": 2,
    "average_price_ghs": 5700.0,
    "min_price_ghs": 2900.0,
    "max_price_ghs": 8500.0,
    "currency": "GHS (Ghana Cedi)",
    "listings": [
      {
        "title": "MacBook Pro 2019",
        "price_ghs": 8500.0,
        "location": "Accra",
        "url": "https://www.ghanabusinessweb.com/classifieds/1254-macbook-pro-2019"
      }
    ]
  }
}
```

### Request Parameters (`POST /api/search`)

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | string | Required | Your question or search topic |
| `top_k` | integer | `50` | Number of ranked results to return (max 500) |
| `categories` | list | Knowledge sources | Defaults to verified knowledge sources: `["news", "academic", "public_services", "wikipedia"]`. Pass `"business"` explicitly to search Ghana Yellow Pages commercial directory. |

*Note: All search endpoints and external providers strictly enforce a hard 4.0-second timeout budget to guarantee rapid responses without hanging.*

---

## AI Inference Frameworks & Tool Calling (vLLM & SGLang)

Ghana Search is built natively for open-weights LLM inference runtimes (**vLLM** and **SGLang**). It provides standard OpenAI-compatible tool specifications and execution endpoints so models (such as Qwen 2.5, Llama 3.1/3.3, Mistral, Command R) can autonomously retrieve verified Ghanaian sources to ground their responses.

### 1. vLLM Tool Calling

Launch vLLM with auto tool calling enabled:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000
```

Autonomous agent loop using the standard OpenAI client:

```python
import requests
from openai import OpenAI

# Connect to local vLLM instance
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 1. Fetch OpenAI-compatible tool definitions from Ghana Search
tools = requests.get("https://search.ghanaopenai.org/api/tools").json()["tools"]

messages = [
    {
        "role": "user",
        "content": "What are the latest VAT invoicing requirements from the Ghana Revenue Authority?",
    }
]

# 2. Model decides to call ghana_search
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

choice = response.choices[0]
if choice.message.tool_calls:
    tool_call = choice.message.tool_calls[0]
    messages.append(choice.message)

    # 3. Execute tool call via Ghana Search tool execution endpoint
    tool_result = requests.post(
        "https://search.ghanaopenai.org/api/tools/execute",
        json={
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,  # Pass raw arguments directly
        },
    ).json()

    # 4. Feed grounded factual records back into conversation history
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result["content"],
    })

    # 5. Generate final referenced answer
    final_response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=messages,
    )
    print(final_response.choices[0].message.content)
```

### 2. SGLang Tool Calling

Launch SGLang server:

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000
```

Autonomous agent execution with SGLang:

```python
import requests
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

# Fetch Ghana Search tool specifications
tools = requests.get("https://search.ghanaopenai.org/api/tools").json()["tools"]

messages = [
    {
        "role": "user",
        "content": "Find recent KNUST research on solar energy battery storage.",
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=messages,
    tools=tools,
)

if response.choices[0].message.tool_calls:
    call = response.choices[0].message.tool_calls[0]
    messages.append(response.choices[0].message)

    # Execute search
    res = requests.post(
        "https://search.ghanaopenai.org/api/tools/execute",
        json={"name": call.function.name, "arguments": call.function.arguments},
    ).json()

    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": res["content"],
    })

    answer = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=messages,
    )
    print(answer.choices[0].message.content)
```

### 3. Model Context Protocol (MCP) Remote Endpoint & Tools

Ghana Search provides three distinct, dedicated MCP tools:

| MCP Tool Name | Functionality | Expected Input | Example Output |
| :--- | :--- | :--- | :--- |
| `ghana_search` | Verified national knowledge (gov, laws, theses, news) | Question or policy topic | Official acts, thesis abstracts, news reports |
| `ghana_business_directory` | Yellow pages & product supplier locator | Product, service, or business name | Matching businesses, locations, and phone numbers |
| `ghana_price_finder` | Classifieds product search & price estimation | Product / item name | Average price (GH₵), price range, active listings |

For MCP clients and agent orchestrators connecting over Server-Sent Events (SSE):
- **SSE Endpoint**: `https://search.ghanaopenai.org/mcp/sse`
- **Tool Schema Endpoint**: `https://search.ghanaopenai.org/api/tools`
- **Tool Execution Endpoint**: `https://search.ghanaopenai.org/api/tools/execute`
- **Commercial Directory Endpoint**: `https://search.ghanaopenai.org/api/business/search`
- **Classifieds Price Finder Endpoint**: `https://search.ghanaopenai.org/api/prices/search`

---

## Offline-First Search & Domain Knowledge Packs

Ghana Search is built from the ground up for **offline-first deployments**. In rural communities, farming districts, remote clinics, and schools without reliable internet, local LLMs (running via vLLM, Ollama, SGLang, or llama.cpp) can query pre-built **Domain Knowledge Packs** with **0ms internet latency and zero network access**.

### 1. Build a Custom Offline Knowledge Pack

You can generate a standalone SQLite knowledge pack from any question list, domain file, or Hugging Face dataset using the built-in pack builder CLI:

```bash
# Build an agricultural offline knowledge pack from Hugging Face:
python3 -m ghana_mcp.builder \
  --hf-dataset ghananlpcommunity/ghana-farmer-qa \
  --column question \
  --limit 500 \
  --output-db data/ghana_farmer_qa_pack.db \
  --export-jsonl data/ghana_farmer_qa_pack.jsonl

# Or build from a local text or CSV file of questions:
python3 -m ghana_mcp.builder \
  --queries-file my_health_questions.txt \
  --output-db data/ghana_health_pack.db
```

The builder runs each question through Ghana Search, retrieves matching publications, university theses, government circulars, and laws, and stores the full article content into a portable SQLite database.

### 2. Run Ghana Search in 100% Offline Mode

Launch the server completely air-gapped without internet access:

```bash
# Launch server pointing to your domain knowledge pack
GHANA_SEARCH_OFFLINE=1 GHANA_SEARCH_DB=data/ghana_farmer_qa_pack.db python3 -m ghana_mcp.server
```

In offline mode:
- **0 external network requests** are performed.
- Queries are matched instantly (< 5ms) against the local knowledge pack using Title-First & Recent-First ranking.
- All MCP tools, REST endpoints (`/api/search`), and tool execution APIs (`/api/tools/execute`) work seamlessly offline for local LLMs.

---

## Offline Corpus and Backup

All permanently cached documents (titles, body text, publication dates, direct PDF URLs, and source metadata) are continuously backed up to the Hugging Face dataset repository:
- Dataset: [https://huggingface.co/datasets/ghanaopenai/ghana-search-corpus](https://huggingface.co/datasets/ghanaopenai/ghana-search-corpus)

---

## Organization

Ghana Search is developed and maintained by **[Ghana Open AI](https://github.com/GhanaOpenAI)** ([https://ghanaopenai.org](https://ghanaopenai.org)).

## License

MIT License. See [LICENSE](LICENSE) for details.
