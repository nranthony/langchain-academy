# MCP (Model Context Protocol) Integration Analysis

## What MCP Brings to the Research Assistant

MCP servers could provide:
- Access to private document repositories
- Specialized academic databases (PubMed, IEEE, ArXiv)
- Internal knowledge bases
- Custom research tools
- RAG systems with vector stores

## How the Architecture Supports MCP Integration

The current design is **extremely well-suited** for adding MCP as a parallel information gathering source:

### 1. Service Layer Makes It Trivial

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPSearchService(BaseService[MCPConfig]):
    """Service for MCP-based information gathering."""

    def __init__(self, config: MCPConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service
        self._session: Optional[ClientSession] = None

    async def connect(self):
        """Connect to MCP server."""
        server_params = StdioServerParameters(
            command=self.config.server_command,
            args=self.config.server_args,
            env=self.config.server_env
        )

        self._session = await stdio_client(server_params).__aenter__()

    async def search_mcp_resources(
        self,
        messages: List[BaseMessage]
    ) -> List[SearchResult]:
        """Search MCP server resources."""
        if not self._session:
            await self.connect()

        # Generate search query from conversation
        query = self.llm_service.invoke_structured(
            SearchQuery,
            [search_instructions] + messages
        ).search_query

        # List available resources from MCP server
        resources = await self._session.list_resources()

        # Read relevant resources
        results = []
        for resource in resources.resources:
            if self._is_relevant(resource, query):
                content = await self._session.read_resource(resource.uri)
                results.append(
                    SearchResult(
                        source="mcp",
                        url=resource.uri,
                        content=content.contents[0].text,
                        relevance_score=None,
                        timestamp=datetime.now().isoformat(),
                        metadata={"mcp_server": self.config.server_name}
                    )
                )

        return results

    async def call_mcp_tool(
        self,
        tool_name: str,
        arguments: dict
    ) -> SearchResult:
        """Call an MCP tool."""
        if not self._session:
            await self.connect()

        result = await self._session.call_tool(tool_name, arguments)

        return SearchResult(
            source="mcp_tool",
            url=f"mcp://{self.config.server_name}/{tool_name}",
            content=str(result.content),
            relevance_score=None,
            timestamp=datetime.now().isoformat(),
            metadata={"tool": tool_name, "args": arguments}
        )
```

### 2. Configuration Layer Handles MCP Servers

```python
@dataclass(kw_only=True)
class MCPConfig(BaseConfig):
    """Configuration for MCP server connections."""

    server_name: str
    server_command: str  # e.g., "node", "python", "npx"
    server_args: List[str]  # e.g., ["server.js"], ["-m", "mcp_server"]
    server_env: Optional[Dict[str, str]] = None
    enabled: bool = True
    timeout: int = 30

@dataclass(kw_only=True)
class SearchConfig(BaseConfig):
    """Configuration for all search services."""

    tavily_api_key: Optional[str] = None
    max_results: int = 3
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    enable_web_search: bool = True
    enable_wikipedia: bool = True

    # MCP configuration
    enable_mcp: bool = True
    mcp_servers: List[MCPConfig] = field(default_factory=list)
```

### 3. Graph Builder Adds MCP Node

```python
class InterviewGraphBuilder(BaseGraphBuilder[InterviewState, ResearchConfig]):
    """Builder for interview subgraph with MCP support."""

    def __init__(
        self,
        config: ResearchConfig,
        interview_service: InterviewService,
        search_service: SearchService,
        mcp_service: Optional[MCPSearchService],  # New!
        report_service: ReportService
    ):
        super().__init__(config)
        self.interview_service = interview_service
        self.search_service = search_service
        self.mcp_service = mcp_service
        self.report_service = report_service

    def create_nodes(self) -> None:
        """Create interview nodes with MCP support."""
        self.builder.add_node("ask_question", self._ask_question_node)
        self.builder.add_node("search_web", self._search_web_node)
        self.builder.add_node("search_wikipedia", self._search_wikipedia_node)

        # Add MCP node if service is available
        if self.mcp_service:
            self.builder.add_node("search_mcp", self._search_mcp_node)

        self.builder.add_node("answer_question", self._answer_question_node)
        self.builder.add_node("save_interview", self._save_interview_node)
        self.builder.add_node("write_section", self._write_section_node)

    def create_edges(self) -> None:
        """Create edges with MCP in parallel."""
        self.builder.add_edge(START, "ask_question")

        # Parallel information gathering (all fire simultaneously)
        self.builder.add_edge("ask_question", "search_web")
        self.builder.add_edge("ask_question", "search_wikipedia")
        if self.mcp_service:
            self.builder.add_edge("ask_question", "search_mcp")  # Parallel with others!

        # All converge to answer_question
        self.builder.add_edge("search_web", "answer_question")
        self.builder.add_edge("search_wikipedia", "answer_question")
        if self.mcp_service:
            self.builder.add_edge("search_mcp", "answer_question")

        # Rest of edges...
        self.builder.add_conditional_edges(
            "answer_question",
            self._route_messages,
            ['ask_question', 'save_interview']
        )
        self.builder.add_edge("save_interview", "write_section")
        self.builder.add_edge("write_section", END)

    async def _search_mcp_node(self, state: InterviewState) -> dict:
        """Node: Search via MCP servers."""
        messages = state["messages"]
        results = await self.mcp_service.search_mcp_resources(messages)
        return {"search_results": results}
```

### 4. Factory Wires It All Together

```python
class ResearchAssistantFactory:
    """Factory for creating research assistant with MCP support."""

    @staticmethod
    async def create(
        config: Optional[ResearchConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> CompiledGraph:
        """Create graph with optional MCP support."""

        if config is None:
            config = ResearchConfig.from_env()

        # Create core services
        llm_service = LLMService(config.llm_config)
        search_service = SearchService(config.search_config, llm_service)
        analyst_service = AnalystService(config, llm_service)
        interview_service = InterviewService(config, llm_service, search_service)
        report_service = ReportService(config, llm_service)

        # Create MCP service if enabled
        mcp_service = None
        if config.search_config.enable_mcp and config.search_config.mcp_servers:
            mcp_service = MCPSearchService(
                config.search_config.mcp_servers[0],  # Or aggregate multiple
                llm_service
            )
            await mcp_service.connect()  # Establish connection

        # Create graph builders with MCP service
        interview_graph_builder = InterviewGraphBuilder(
            config,
            interview_service,
            search_service,
            mcp_service,  # Injected!
            report_service
        )

        research_graph_builder = ResearchGraphBuilder(
            config,
            analyst_service,
            interview_graph_builder,
            report_service
        )

        graph = research_graph_builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_feedback"]
        )

        return graph
```

## Why This Architecture Excels for MCP Integration

### ✅ 1. Parallel Execution by Design
- The graph already has parallel edges from `ask_question` → multiple search nodes
- Adding `search_mcp` as another parallel node is trivial
- State aggregation via `Annotated[List[SearchResult], operator.add]` automatically merges results

### ✅ 2. Service Layer Encapsulation
- MCP complexity is hidden in `MCPSearchService`
- Graph nodes remain simple wrappers
- Easy to swap MCP implementations or add multiple MCP servers

### ✅ 3. Optional Dependency Pattern
- MCP service is `Optional[MCPSearchService]`
- Graph builder conditionally adds MCP nodes if service exists
- Graceful degradation if MCP unavailable

### ✅ 4. Structured State with Metadata
- `SearchResult` includes `source="mcp"` and `metadata` fields
- Report service can cite MCP sources differently
- Can track which MCP server provided which information

### ✅ 5. Multiple MCP Servers Support

```python
class AggregatedMCPService(BaseService[SearchConfig]):
    """Service managing multiple MCP servers."""

    def __init__(self, config: SearchConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service
        self.mcp_services: List[MCPSearchService] = []

    async def connect_all(self):
        """Connect to all configured MCP servers."""
        for mcp_config in self.config.mcp_servers:
            service = MCPSearchService(mcp_config, self.llm_service)
            await service.connect()
            self.mcp_services.append(service)

    async def search_all_servers(
        self,
        messages: List[BaseMessage]
    ) -> List[SearchResult]:
        """Search across all MCP servers concurrently."""
        tasks = [
            service.search_mcp_resources(messages)
            for service in self.mcp_services
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten and filter errors
        all_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning("mcp_search_failed", error=str(result))
            else:
                all_results.extend(result)

        return all_results
```

## Example Usage with MCP

```python
from research_assistant.app import ResearchAssistantFactory
from research_assistant.config import ResearchConfig, SearchConfig, MCPConfig

# Configure MCP servers
config = ResearchConfig(
    search_config=SearchConfig(
        enable_web_search=True,
        enable_wikipedia=True,
        enable_mcp=True,
        mcp_servers=[
            MCPConfig(
                server_name="pubmed",
                server_command="npx",
                server_args=["-y", "@modelcontextprotocol/server-pubmed"]
            ),
            MCPConfig(
                server_name="arxiv",
                server_command="python",
                server_args=["-m", "mcp_server_arxiv"],
                server_env={"ARXIV_API_KEY": "..."}
            )
        ]
    )
)

# Create graph with MCP support
graph = await ResearchAssistantFactory.create(config)

# All three sources (web, wikipedia, MCP) work in parallel
result = await graph.ainvoke({
    "topic": "CRISPR gene therapy efficacy",
    "max_analysts": 3
})
```

## Architecture Diagram with MCP

```
Interview Graph (with MCP):

    ask_question
         |
    ┌────┼────┬────────┐
    ↓    ↓    ↓        ↓
search_web │ search_mcp
    │  search_wikipedia
    │    │    │        │
    └────┼────┴────────┘
         ↓
   answer_question
    (receives aggregated
     SearchResult[] from
     all sources)
```

## Updated Summary of Changes Table

| Change | Priority | Gemini Suggested | Rationale |
|--------|----------|------------------|-----------|
| Rename to "Layered Architecture" | High | ✅ Yes | Accurate terminology |
| Remove `BaseService.process()` | High | ✅ Yes | Eliminate unused abstraction |
| Add async support | High | ✅ Yes | **Critical for MCP** (async protocol) |
| Improve state schema | High | ✅ Yes | **Essential for MCP** (source tracking) |
| **Add MCP integration support** | **High** | **✅ Yes** | **MCP as parallel information source; requires async, structured state, and service layer - all present in design** |
| Add state versioning | Medium | ❌ No | Handle schema evolution |
| Enhance observability | Medium | ❌ No | Production readiness |
| Add plugin architecture | Low | ❌ No | Future extensibility (note: MCP essentially provides this) |

## Key Insight

The architecture is **exceptionally well-positioned for MCP integration** because:

1. **Async support** (from Gemini's feedback) is required for MCP's async protocol
2. **Structured state with metadata** (from Gemini's feedback) is perfect for tracking MCP sources
3. **Service layer** makes adding `MCPSearchService` clean and isolated
4. **Parallel graph edges** already exist for multiple search sources
5. **Optional dependencies** pattern allows graceful MCP unavailability

**In fact, adding MCP is exactly the extensibility scenario the architecture was designed for.** It requires:
- ✅ New config dataclass (`MCPConfig`)
- ✅ New service (`MCPSearchService`)
- ✅ New node in builder (`_search_mcp_node`)
- ✅ New edge to parallel search nodes
- ✅ Zero changes to existing code

This is textbook extensibility through composition and dependency injection. The MCP integration validates the architecture's design quality.

## Implementation Checklist

### Phase 1: Add MCP Configuration
- [ ] Create `MCPConfig` dataclass in `config/mcp.py`
- [ ] Add `enable_mcp` and `mcp_servers` to `SearchConfig`
- [ ] Support environment variable configuration for MCP servers

### Phase 2: Create MCP Service
- [ ] Create `services/mcp.py` with `MCPSearchService`
- [ ] Implement `connect()` for establishing MCP connections
- [ ] Implement `search_mcp_resources()` for resource retrieval
- [ ] Implement `call_mcp_tool()` for tool invocation
- [ ] Add error handling and retry logic

### Phase 3: Update Graph Builder
- [ ] Add `mcp_service` parameter to `InterviewGraphBuilder.__init__`
- [ ] Add conditional `search_mcp` node in `create_nodes()`
- [ ] Add parallel edge from `ask_question` to `search_mcp` in `create_edges()`
- [ ] Add edge from `search_mcp` to `answer_question`

### Phase 4: Update Factory
- [ ] Modify `ResearchAssistantFactory.create()` to instantiate MCP service
- [ ] Add MCP connection logic
- [ ] Inject MCP service into `InterviewGraphBuilder`
- [ ] Handle MCP unavailability gracefully

### Phase 5: Testing
- [ ] Unit tests for `MCPSearchService`
- [ ] Integration tests with mock MCP server
- [ ] Test parallel execution of web, wikipedia, and MCP searches
- [ ] Test graceful degradation when MCP unavailable
- [ ] Test multiple MCP server support

### Phase 6: Documentation
- [ ] Document MCP configuration options
- [ ] Provide example MCP server configurations
- [ ] Update architecture diagrams
- [ ] Add troubleshooting guide for MCP connections
