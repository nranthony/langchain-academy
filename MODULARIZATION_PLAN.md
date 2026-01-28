# Research Assistant Modularization Plan

## Executive Summary

This plan outlines a modular Python architecture for scaling the research assistant LangGraph application. The design eliminates global variables, promotes reusability, improves testability, and enables production deployment through dependency injection, composition patterns, and clear separation of concerns.

---

## Current Issues Analysis

### Problems Identified

1. **Global LLM Object**: `llm = ChatOpenAI(model="gpt-5-nano", temperature=0)` defined at module level
2. **Global State Graphs**: Graph builders and compiled graphs exist at module level
3. **No Dependency Injection**: Functions don't receive LLM or configuration objects
4. **Repetitive Function Patterns**: Similar node function structures repeated across the codebase
5. **Tight Coupling**: Nodes implicitly depend on module-level variables
6. **Limited Testability**: Hard to test nodes in isolation due to implicit dependencies
7. **Configuration Hardcoding**: No centralized configuration management

### Well-Designed Elements to Preserve

- **TypedDict State Classes**: Clear state contracts (`GenerateAnalystsState`, `InterviewState`, `ResearchGraphState`)
- **Pydantic Models**: Well-defined schemas (`Analyst`, `Perspectives`, `SearchQuery`)
- **Functional Node Design**: Stateless node functions with clear inputs/outputs
- **Subgraph Pattern**: Interview graph as reusable component

---

## Proposed Architecture

### Core Design Principles

1. **Dependency Injection**: All dependencies (LLM, config, store) explicitly passed
2. **Builder Pattern**: Factory classes construct graphs with injected dependencies
3. **Configuration as Code**: Dataclass-based configuration management
4. **Composition**: Graphs composed from reusable node factories
5. **Interface Segregation**: Clear interfaces for nodes, graphs, and services
6. **Single Responsibility**: Each class has one focused purpose

---

## Base Pattern: Service-Oriented Architecture with Builder Pattern

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                  Application Layer                       │
│  (graph execution, CLI, API endpoints)                  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                   Graph Layer                            │
│  (GraphBuilder classes: ResearchGraphBuilder,           │
│   InterviewGraphBuilder)                                │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│                   Service Layer                          │
│  (AnalystService, SearchService, ReportService)         │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│               Configuration Layer                        │
│  (ResearchConfig, LLMConfig, SearchConfig)              │
└─────────────────────────────────────────────────────────┘
```

---

## Base Classes Design

### 1. Configuration Management

#### 1.1 Base Configuration Class

```python
@dataclass(kw_only=True)
class BaseConfig:
    """Base configuration with environment variable support."""

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "BaseConfig":
        """Create configuration from LangGraph RunnableConfig."""
        pass

    @classmethod
    def from_env(cls) -> "BaseConfig":
        """Create configuration from environment variables."""
        pass

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        pass
```

**Purpose**: Eliminate hardcoded values, support multiple configuration sources

**Usage**: All specific configs inherit from this

#### 1.2 LLM Configuration

```python
@dataclass(kw_only=True)
class LLMConfig(BaseConfig):
    """Configuration for language models."""

    model_name: str = "gpt-5-nano"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 60
    api_key: Optional[str] = None

    def create_llm(self) -> ChatOpenAI:
        """Factory method to create LLM instance."""
        pass
```

**Purpose**: Centralize LLM configuration, enable easy swapping of models

**Benefits**:
- Test with different models without code changes
- Inject mock LLMs for testing
- Support multiple LLM configurations in same application

#### 1.3 Search Configuration

```python
@dataclass(kw_only=True)
class SearchConfig(BaseConfig):
    """Configuration for search services."""

    tavily_api_key: Optional[str] = None
    max_results: int = 3
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    enable_web_search: bool = True
    enable_wikipedia: bool = True
```

**Purpose**: Centralize search configuration

**Benefits**: Easy to disable/enable search sources, configure domain filtering

#### 1.4 Research Configuration

```python
@dataclass(kw_only=True)
class ResearchConfig(BaseConfig):
    """Master configuration for research assistant."""

    llm_config: LLMConfig = field(default_factory=LLMConfig)
    search_config: SearchConfig = field(default_factory=SearchConfig)
    max_analysts: int = 3
    max_interview_turns: int = 2
    user_id: str = "default-user"
    session_id: Optional[str] = None

    # Prompt templates (externalized)
    analyst_instructions_template: str = "..."
    question_instructions_template: str = "..."
    answer_instructions_template: str = "..."
    section_writer_instructions_template: str = "..."
    report_writer_instructions_template: str = "..."
    intro_conclusion_instructions_template: str = "..."
```

**Purpose**: Master configuration composing all sub-configs

**Benefits**: Single source of truth, easy to override specific settings

---

### 2. Service Layer

#### 2.1 Base Service Interface

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TState = TypeVar('TState')
TConfig = TypeVar('TConfig', bound=BaseConfig)

class BaseService(ABC, Generic[TState, TConfig]):
    """Base class for all services that interact with LangGraph nodes."""

    def __init__(self, config: TConfig):
        self.config = config

    @abstractmethod
    def process(self, state: TState, **kwargs) -> dict:
        """Process state and return updates."""
        pass
```

**Purpose**: Standardize service interface

**Benefits**: All services follow same pattern, easy to test and compose

#### 2.2 LLM Service

```python
class LLMService(BaseService[MessagesState, LLMConfig]):
    """Service for LLM interactions with dependency injection."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        """Lazy-loaded LLM instance."""
        if self._llm is None:
            self._llm = self.config.create_llm()
        return self._llm

    def invoke_with_system_message(
        self,
        system_message: str,
        messages: List[BaseMessage]
    ) -> BaseMessage:
        """Invoke LLM with system message prefix."""
        pass

    def invoke_structured(
        self,
        output_schema: Type[BaseModel],
        messages: List[BaseMessage]
    ) -> BaseModel:
        """Invoke LLM with structured output."""
        pass
```

**Purpose**: Encapsulate LLM interactions, enable dependency injection

**Benefits**:
- Single place to manage LLM lifecycle
- Lazy loading for efficiency
- Easy to mock for testing
- Structured output abstraction

#### 2.3 Analyst Service

```python
class AnalystService(BaseService[GenerateAnalystsState, ResearchConfig]):
    """Service for creating and managing analysts."""

    def __init__(self, config: ResearchConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service

    def create_analysts(
        self,
        topic: str,
        max_analysts: int,
        human_feedback: Optional[str] = None
    ) -> List[Analyst]:
        """Create analyst personas for topic."""
        pass

    def format_analyst_instructions(
        self,
        topic: str,
        max_analysts: int,
        human_feedback: str
    ) -> str:
        """Format analyst creation prompt."""
        pass
```

**Purpose**: Business logic for analyst creation

**Benefits**: Reusable across different graph configurations

#### 2.4 Search Service

```python
class SearchService(BaseService[InterviewState, SearchConfig]):
    """Service for web and Wikipedia search."""

    def __init__(self, config: SearchConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service
        self._tavily_search: Optional[TavilySearch] = None

    @property
    def tavily_search(self) -> TavilySearch:
        """Lazy-loaded Tavily search instance."""
        pass

    def search_web(
        self,
        messages: List[BaseMessage]
    ) -> List[str]:
        """Search web and return formatted documents."""
        pass

    def search_wikipedia(
        self,
        messages: List[BaseMessage]
    ) -> List[str]:
        """Search Wikipedia and return formatted documents."""
        pass

    def generate_search_query(
        self,
        messages: List[BaseMessage]
    ) -> str:
        """Generate search query from conversation."""
        pass

    def format_search_results(
        self,
        results: List[dict],
        source_type: str
    ) -> List[str]:
        """Format search results with proper document tags."""
        pass
```

**Purpose**: Encapsulate search logic, support multiple search backends

**Benefits**:
- Easy to add new search sources
- Centralized result formatting
- Mock-friendly for testing

#### 2.5 Interview Service

```python
class InterviewService(BaseService[InterviewState, ResearchConfig]):
    """Service for conducting analyst-expert interviews."""

    def __init__(
        self,
        config: ResearchConfig,
        llm_service: LLMService,
        search_service: SearchService
    ):
        super().__init__(config)
        self.llm_service = llm_service
        self.search_service = search_service

    def generate_question(
        self,
        analyst: Analyst,
        messages: List[BaseMessage]
    ) -> BaseMessage:
        """Generate analyst question."""
        pass

    def generate_answer(
        self,
        analyst: Analyst,
        messages: List[BaseMessage],
        context: List[str]
    ) -> BaseMessage:
        """Generate expert answer using context."""
        pass

    def format_interview(
        self,
        messages: List[BaseMessage]
    ) -> str:
        """Convert messages to interview transcript."""
        pass

    def should_continue_interview(
        self,
        messages: List[BaseMessage],
        max_turns: int
    ) -> bool:
        """Determine if interview should continue."""
        pass
```

**Purpose**: Interview orchestration logic

**Benefits**: Reusable interview logic, testable in isolation

#### 2.6 Report Service

```python
class ReportService(BaseService[ResearchGraphState, ResearchConfig]):
    """Service for generating research reports."""

    def __init__(self, config: ResearchConfig, llm_service: LLMService):
        super().__init__(config)
        self.llm_service = llm_service

    def write_section(
        self,
        analyst: Analyst,
        interview: str,
        context: List[str]
    ) -> str:
        """Write report section from interview."""
        pass

    def write_report_body(
        self,
        sections: List[str],
        topic: str
    ) -> str:
        """Consolidate sections into report body."""
        pass

    def write_introduction(
        self,
        sections: List[str],
        topic: str
    ) -> str:
        """Write report introduction."""
        pass

    def write_conclusion(
        self,
        sections: List[str],
        topic: str
    ) -> str:
        """Write report conclusion."""
        pass

    def finalize_report(
        self,
        introduction: str,
        content: str,
        conclusion: str
    ) -> str:
        """Assemble final report."""
        pass
```

**Purpose**: Report generation and assembly

**Benefits**: Clear separation of report writing concerns

---

### 3. Graph Builder Layer

#### 3.1 Base Graph Builder

```python
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TState = TypeVar('TState')
TConfig = TypeVar('TConfig', bound=BaseConfig)

class BaseGraphBuilder(ABC, Generic[TState, TConfig]):
    """Base class for building LangGraph StateGraphs."""

    def __init__(self, config: TConfig):
        self.config = config
        self.builder: Optional[StateGraph] = None

    @abstractmethod
    def create_nodes(self) -> None:
        """Create and add nodes to graph."""
        pass

    @abstractmethod
    def create_edges(self) -> None:
        """Create and add edges to graph."""
        pass

    def build(self) -> StateGraph:
        """Build the complete graph."""
        if self.builder is None:
            self.builder = StateGraph(self._get_state_class())
            self.create_nodes()
            self.create_edges()
        return self.builder

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[List[str]] = None,
        interrupt_after: Optional[List[str]] = None
    ) -> CompiledGraph:
        """Compile the graph with optional checkpointing and interrupts."""
        graph = self.build()
        return graph.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after
        )

    @abstractmethod
    def _get_state_class(self) -> type:
        """Return the state class for this graph."""
        pass
```

**Purpose**: Standardize graph building process

**Benefits**: Consistent graph creation, reusable compilation logic

#### 3.2 Interview Graph Builder

```python
class InterviewGraphBuilder(BaseGraphBuilder[InterviewState, ResearchConfig]):
    """Builder for interview subgraph."""

    def __init__(
        self,
        config: ResearchConfig,
        interview_service: InterviewService,
        search_service: SearchService,
        report_service: ReportService
    ):
        super().__init__(config)
        self.interview_service = interview_service
        self.search_service = search_service
        self.report_service = report_service

    def _get_state_class(self) -> type:
        return InterviewState

    def create_nodes(self) -> None:
        """Create interview nodes with dependency injection."""
        self.builder.add_node("ask_question", self._ask_question_node)
        self.builder.add_node("search_web", self._search_web_node)
        self.builder.add_node("search_wikipedia", self._search_wikipedia_node)
        self.builder.add_node("answer_question", self._answer_question_node)
        self.builder.add_node("save_interview", self._save_interview_node)
        self.builder.add_node("write_section", self._write_section_node)

    def create_edges(self) -> None:
        """Create interview graph edges."""
        self.builder.add_edge(START, "ask_question")
        self.builder.add_edge("ask_question", "search_web")
        self.builder.add_edge("ask_question", "search_wikipedia")
        self.builder.add_edge("search_web", "answer_question")
        self.builder.add_edge("search_wikipedia", "answer_question")
        self.builder.add_conditional_edges(
            "answer_question",
            self._route_messages,
            ['ask_question', 'save_interview']
        )
        self.builder.add_edge("save_interview", "write_section")
        self.builder.add_edge("write_section", END)

    # Node factory methods with dependency injection
    def _ask_question_node(self, state: InterviewState) -> dict:
        """Node: Generate analyst question."""
        analyst = state["analyst"]
        messages = state["messages"]
        question = self.interview_service.generate_question(analyst, messages)
        return {"messages": [question]}

    def _search_web_node(self, state: InterviewState) -> dict:
        """Node: Search web."""
        messages = state["messages"]
        results = self.search_service.search_web(messages)
        return {"context": results}

    def _search_wikipedia_node(self, state: InterviewState) -> dict:
        """Node: Search Wikipedia."""
        messages = state["messages"]
        results = self.search_service.search_wikipedia(messages)
        return {"context": results}

    def _answer_question_node(self, state: InterviewState) -> dict:
        """Node: Generate expert answer."""
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]
        answer = self.interview_service.generate_answer(analyst, messages, context)
        answer.name = "expert"
        return {"messages": [answer]}

    def _save_interview_node(self, state: InterviewState) -> dict:
        """Node: Save interview transcript."""
        messages = state["messages"]
        interview = self.interview_service.format_interview(messages)
        return {"interview": interview}

    def _write_section_node(self, state: InterviewState) -> dict:
        """Node: Write report section."""
        analyst = state["analyst"]
        interview = state["interview"]
        context = state["context"]
        section = self.report_service.write_section(analyst, interview, context)
        return {"sections": [section]}

    def _route_messages(
        self,
        state: InterviewState
    ) -> Literal["ask_question", "save_interview"]:
        """Conditional edge: Route between question and save."""
        messages = state["messages"]
        max_turns = state.get("max_num_turns", self.config.max_interview_turns)

        if self.interview_service.should_continue_interview(messages, max_turns):
            return "ask_question"
        return "save_interview"
```

**Purpose**: Build interview subgraph with injected services

**Benefits**:
- All dependencies explicit
- Node functions are methods with access to services
- Easy to test node logic
- Clear separation between graph structure and business logic

#### 3.3 Research Graph Builder

```python
class ResearchGraphBuilder(BaseGraphBuilder[ResearchGraphState, ResearchConfig]):
    """Builder for main research graph."""

    def __init__(
        self,
        config: ResearchConfig,
        analyst_service: AnalystService,
        interview_graph_builder: InterviewGraphBuilder,
        report_service: ReportService
    ):
        super().__init__(config)
        self.analyst_service = analyst_service
        self.interview_graph_builder = interview_graph_builder
        self.report_service = report_service

    def _get_state_class(self) -> type:
        return ResearchGraphState

    def create_nodes(self) -> None:
        """Create research graph nodes."""
        self.builder.add_node("create_analysts", self._create_analysts_node)
        self.builder.add_node("human_feedback", self._human_feedback_node)

        # Compile interview subgraph as node
        interview_graph = self.interview_graph_builder.compile()
        self.builder.add_node("conduct_interview", interview_graph)

        self.builder.add_node("write_report", self._write_report_node)
        self.builder.add_node("write_introduction", self._write_introduction_node)
        self.builder.add_node("write_conclusion", self._write_conclusion_node)
        self.builder.add_node("finalize_report", self._finalize_report_node)

    def create_edges(self) -> None:
        """Create research graph edges."""
        self.builder.add_edge(START, "create_analysts")
        self.builder.add_edge("create_analysts", "human_feedback")
        self.builder.add_conditional_edges(
            "human_feedback",
            self._initiate_all_interviews,
            ["create_analysts", "conduct_interview"]
        )
        self.builder.add_edge("conduct_interview", "write_report")
        self.builder.add_edge("conduct_interview", "write_introduction")
        self.builder.add_edge("conduct_interview", "write_conclusion")
        self.builder.add_edge(
            ["write_conclusion", "write_report", "write_introduction"],
            "finalize_report"
        )
        self.builder.add_edge("finalize_report", END)

    # Node factory methods
    def _create_analysts_node(self, state: ResearchGraphState) -> dict:
        """Node: Create analyst personas."""
        topic = state["topic"]
        max_analysts = state["max_analysts"]
        human_feedback = state.get("human_analyst_feedback", "")

        analysts = self.analyst_service.create_analysts(
            topic,
            max_analysts,
            human_feedback
        )
        return {"analysts": analysts}

    def _human_feedback_node(self, state: ResearchGraphState) -> dict:
        """Node: No-op for interrupt."""
        pass

    def _write_report_node(self, state: ResearchGraphState) -> dict:
        """Node: Write report body."""
        sections = state["sections"]
        topic = state["topic"]
        content = self.report_service.write_report_body(sections, topic)
        return {"content": content}

    def _write_introduction_node(self, state: ResearchGraphState) -> dict:
        """Node: Write introduction."""
        sections = state["sections"]
        topic = state["topic"]
        introduction = self.report_service.write_introduction(sections, topic)
        return {"introduction": introduction}

    def _write_conclusion_node(self, state: ResearchGraphState) -> dict:
        """Node: Write conclusion."""
        sections = state["sections"]
        topic = state["topic"]
        conclusion = self.report_service.write_conclusion(sections, topic)
        return {"conclusion": conclusion}

    def _finalize_report_node(self, state: ResearchGraphState) -> dict:
        """Node: Assemble final report."""
        introduction = state["introduction"]
        content = state["content"]
        conclusion = state["conclusion"]
        final_report = self.report_service.finalize_report(
            introduction,
            content,
            conclusion
        )
        return {"final_report": final_report}

    def _initiate_all_interviews(
        self,
        state: ResearchGraphState
    ) -> Union[str, List[Send]]:
        """Conditional edge: Start interviews or return to create_analysts."""
        human_feedback = state.get("human_analyst_feedback", "approve")

        if human_feedback.lower() != "approve":
            return "create_analysts"

        topic = state["topic"]
        analysts = state["analysts"]

        # Parallel execution via Send() API
        return [
            Send(
                "conduct_interview",
                {
                    "analyst": analyst,
                    "messages": [
                        HumanMessage(
                            content=f"So you said you were writing an article on {topic}?"
                        )
                    ]
                }
            )
            for analyst in analysts
        ]
```

**Purpose**: Build main research graph composing subgraphs

**Benefits**:
- Clear composition of interview subgraph
- All services injected
- Parallelization logic cleanly separated

---

### 4. Application Layer

#### 4.1 Research Assistant Factory

```python
class ResearchAssistantFactory:
    """Factory for creating research assistant graph with all dependencies."""

    @staticmethod
    def create(
        config: Optional[ResearchConfig] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> CompiledGraph:
        """Create fully configured research assistant graph."""

        # 1. Configuration
        if config is None:
            config = ResearchConfig.from_env()

        # 2. Create services
        llm_service = LLMService(config.llm_config)
        search_service = SearchService(config.search_config, llm_service)
        analyst_service = AnalystService(config, llm_service)
        interview_service = InterviewService(config, llm_service, search_service)
        report_service = ReportService(config, llm_service)

        # 3. Create graph builders
        interview_graph_builder = InterviewGraphBuilder(
            config,
            interview_service,
            search_service,
            report_service
        )

        research_graph_builder = ResearchGraphBuilder(
            config,
            analyst_service,
            interview_graph_builder,
            report_service
        )

        # 4. Compile graph
        graph = research_graph_builder.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_feedback"]
        )

        return graph

    @staticmethod
    def create_from_config_file(
        config_path: str,
        checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> CompiledGraph:
        """Create graph from configuration file."""
        # Load config from YAML/JSON
        config = ResearchConfig.from_file(config_path)
        return ResearchAssistantFactory.create(config, checkpointer)
```

**Purpose**: Single entry point for creating configured graphs

**Benefits**:
- Hides complexity of dependency wiring
- Easy to create graph for different contexts (testing, production)
- Centralized configuration loading

#### 4.2 Research Assistant Runner

```python
class ResearchAssistantRunner:
    """High-level interface for running research assistant."""

    def __init__(
        self,
        graph: CompiledGraph,
        config: ResearchConfig
    ):
        self.graph = graph
        self.config = config

    def run_research(
        self,
        topic: str,
        max_analysts: Optional[int] = None,
        thread_id: Optional[str] = None
    ) -> str:
        """Run complete research process with human-in-the-loop."""

        # Initialize thread
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        thread = {"configurable": {"thread_id": thread_id}}

        # Determine max analysts
        if max_analysts is None:
            max_analysts = self.config.max_analysts

        # Initial input
        initial_state = {
            "topic": topic,
            "max_analysts": max_analysts
        }

        # Run until first interrupt
        for event in self.graph.stream(initial_state, thread, stream_mode="values"):
            if "analysts" in event:
                yield {"event": "analysts_created", "analysts": event["analysts"]}

        # Wait for human feedback (handled externally)
        yield {"event": "awaiting_feedback", "thread_id": thread_id}

    def provide_feedback(
        self,
        thread_id: str,
        feedback: Optional[str]
    ) -> None:
        """Provide human feedback and continue execution."""
        thread = {"configurable": {"thread_id": thread_id}}

        # Update state with feedback
        self.graph.update_state(
            thread,
            {"human_analyst_feedback": feedback},
            as_node="human_feedback"
        )

    def continue_research(
        self,
        thread_id: str
    ) -> Generator[dict, None, None]:
        """Continue research after feedback."""
        thread = {"configurable": {"thread_id": thread_id}}

        # Stream remaining execution
        for event in self.graph.stream(None, thread, stream_mode="updates"):
            node_name = next(iter(event.keys()))
            yield {"event": "node_completed", "node": node_name}

        # Get final report
        final_state = self.graph.get_state(thread)
        final_report = final_state.values.get("final_report")

        yield {"event": "research_complete", "report": final_report}
```

**Purpose**: High-level orchestration of research workflow

**Benefits**:
- Hides LangGraph execution details
- Clean API for CLI/web interface
- Generator pattern for streaming updates

---

## Directory Structure

```
research_assistant/
│
├── config/
│   ├── __init__.py
│   ├── base.py              # BaseConfig
│   ├── llm.py               # LLMConfig
│   ├── search.py            # SearchConfig
│   └── research.py          # ResearchConfig
│
├── models/
│   ├── __init__.py
│   ├── analyst.py           # Analyst, Perspectives
│   ├── state.py             # State classes (GenerateAnalystsState, etc.)
│   └── search.py            # SearchQuery
│
├── services/
│   ├── __init__.py
│   ├── base.py              # BaseService
│   ├── llm.py               # LLMService
│   ├── analyst.py           # AnalystService
│   ├── search.py            # SearchService
│   ├── interview.py         # InterviewService
│   └── report.py            # ReportService
│
├── graphs/
│   ├── __init__.py
│   ├── base.py              # BaseGraphBuilder
│   ├── interview.py         # InterviewGraphBuilder
│   └── research.py          # ResearchGraphBuilder
│
├── prompts/
│   ├── __init__.py
│   ├── analyst.py           # Analyst prompt templates
│   ├── interview.py         # Interview prompt templates
│   └── report.py            # Report prompt templates
│
├── app/
│   ├── __init__.py
│   ├── factory.py           # ResearchAssistantFactory
│   └── runner.py            # ResearchAssistantRunner
│
├── utils/
│   ├── __init__.py
│   └── formatting.py        # Document formatting utilities
│
└── tests/
    ├── unit/
    │   ├── test_services.py
    │   ├── test_graphs.py
    │   └── test_config.py
    ├── integration/
    │   └── test_workflows.py
    └── fixtures/
        └── mock_data.py
```

---

## Migration Path

### Phase 1: Extract Configuration (Week 1)

**Tasks:**
1. Create `config/` module with all config classes
2. Replace hardcoded values with config references
3. Test configuration loading from environment

**Files to Create:**
- `config/base.py`
- `config/llm.py`
- `config/search.py`
- `config/research.py`

**Expected Outcome**: No global variables, all configuration centralized

---

### Phase 2: Extract Services (Week 2)

**Tasks:**
1. Create `services/` module with service classes
2. Move business logic from node functions to service methods
3. Add unit tests for each service

**Files to Create:**
- `services/base.py`
- `services/llm.py`
- `services/analyst.py`
- `services/search.py`
- `services/interview.py`
- `services/report.py`

**Expected Outcome**: Reusable, testable business logic layer

---

### Phase 3: Refactor Graph Builders (Week 3)

**Tasks:**
1. Create `graphs/` module with builder classes
2. Convert node functions to builder methods
3. Inject services into builders

**Files to Create:**
- `graphs/base.py`
- `graphs/interview.py`
- `graphs/research.py`

**Expected Outcome**: Composable, dependency-injected graphs

---

### Phase 4: Application Layer (Week 4)

**Tasks:**
1. Create `app/` module with factory and runner
2. Update notebook to use factory
3. Create CLI using runner
4. Add integration tests

**Files to Create:**
- `app/factory.py`
- `app/runner.py`
- `cli.py`

**Expected Outcome**: Clean interfaces for using research assistant

---

### Phase 5: Externalize Prompts (Week 5)

**Tasks:**
1. Create `prompts/` module
2. Move prompt templates to dedicated files
3. Support loading from external files (YAML/JSON)

**Files to Create:**
- `prompts/analyst.py`
- `prompts/interview.py`
- `prompts/report.py`

**Expected Outcome**: Easily modifiable prompts without code changes

---

## Testing Strategy

### Unit Tests

**Service Tests:**
```python
# Example: Testing AnalystService
def test_create_analysts():
    # Arrange
    mock_llm_service = MockLLMService()
    config = ResearchConfig(max_analysts=3)
    service = AnalystService(config, mock_llm_service)

    # Act
    analysts = service.create_analysts("AI Safety", 3)

    # Assert
    assert len(analysts) == 3
    assert all(isinstance(a, Analyst) for a in analysts)
```

**Graph Builder Tests:**
```python
# Example: Testing InterviewGraphBuilder
def test_interview_graph_structure():
    # Arrange
    config = ResearchConfig()
    services = create_mock_services(config)
    builder = InterviewGraphBuilder(config, *services)

    # Act
    graph = builder.build()

    # Assert
    assert "ask_question" in graph.nodes
    assert "search_web" in graph.nodes
    assert graph.edges["ask_question"]["search_web"] is not None
```

### Integration Tests

**End-to-End Workflow:**
```python
def test_complete_research_workflow():
    # Use in-memory checkpointer and mock LLM
    config = ResearchConfig(
        llm_config=LLMConfig(model_name="mock-model")
    )
    graph = ResearchAssistantFactory.create(config, MemorySaver())
    runner = ResearchAssistantRunner(graph, config)

    # Run research
    events = list(runner.run_research("AI Safety", max_analysts=2))

    # Verify flow
    assert any(e["event"] == "analysts_created" for e in events)
    assert any(e["event"] == "research_complete" for e in events)
```

---

## Performance Considerations

### Lazy Loading

- LLM instances created only when needed
- Search services instantiated on first use
- Configuration loaded once per execution

### Caching

```python
# Example: Add caching to SearchService
from functools import lru_cache

class SearchService:
    @lru_cache(maxsize=100)
    def search_web(self, query: str) -> List[str]:
        """Cached web search."""
        pass
```

### Parallelization

- Send() API already enables parallel interviews
- Consider async/await for search operations
- Use multiprocessing for CPU-bound report generation

---

## Production Readiness Checklist

### Observability

- [ ] Add logging to all services (using `structlog`)
- [ ] Instrument with metrics (Prometheus/StatsD)
- [ ] Add distributed tracing (OpenTelemetry)
- [ ] LangSmith integration for LLM call tracing

### Error Handling

- [ ] Add retry logic to LLM calls (using `tenacity`)
- [ ] Graceful degradation for search failures
- [ ] Validation of all user inputs
- [ ] Custom exception hierarchy

### Security

- [ ] API key management (never in code)
- [ ] Input sanitization for prompts
- [ ] Rate limiting for external API calls
- [ ] Audit logging for all operations

### Scalability

- [ ] Replace MemorySaver with persistent checkpointer (Redis/PostgreSQL)
- [ ] Add connection pooling for external APIs
- [ ] Implement circuit breakers for unstable services
- [ ] Add request queuing for high load

### Documentation

- [ ] API documentation (Sphinx/MkDocs)
- [ ] Architecture diagrams
- [ ] Deployment guides
- [ ] Troubleshooting runbooks

---

## Example Usage After Refactoring

### Simple Usage

```python
from research_assistant.app import ResearchAssistantFactory

# Create with defaults
graph = ResearchAssistantFactory.create()

# Run research
result = graph.invoke({
    "topic": "AI Safety",
    "max_analysts": 3
})
```

### Advanced Usage

```python
from research_assistant.config import ResearchConfig, LLMConfig, SearchConfig
from research_assistant.app import ResearchAssistantFactory, ResearchAssistantRunner
from langgraph.checkpoint.postgres import PostgresSaver

# Custom configuration
config = ResearchConfig(
    llm_config=LLMConfig(
        model_name="gpt-5-nano",
        temperature=0.7
    ),
    search_config=SearchConfig(
        max_results=5,
        include_domains=["arxiv.org", "scholar.google.com"]
    ),
    max_analysts=4,
    max_interview_turns=3
)

# Production checkpointer
checkpointer = PostgresSaver.from_conn_string("postgresql://...")

# Create and run
graph = ResearchAssistantFactory.create(config, checkpointer)
runner = ResearchAssistantRunner(graph, config)

# Execute with streaming
for event in runner.run_research("Quantum Computing", max_analysts=4):
    if event["event"] == "analysts_created":
        print(f"Created {len(event['analysts'])} analysts")
    elif event["event"] == "research_complete":
        print(event["report"])
```

### Testing Usage

```python
from research_assistant.app import ResearchAssistantFactory
from research_assistant.config import ResearchConfig, LLMConfig
from unittest.mock import Mock

# Mock configuration for testing
config = ResearchConfig(
    llm_config=LLMConfig(model_name="mock-model")
)

# Inject mock LLM service (would need to add this to factory)
mock_llm = Mock()
graph = ResearchAssistantFactory.create_with_mocks(
    config=config,
    llm_service=mock_llm
)

# Test without making real API calls
result = graph.invoke({"topic": "Test Topic", "max_analysts": 2})
```

---

## Benefits Summary

### Before Refactoring

❌ Global variables (`llm`, `graph`)
❌ Implicit dependencies
❌ Hard to test
❌ Hard to configure
❌ Tight coupling
❌ Not production-ready
❌ Can't swap implementations

### After Refactoring

✅ Explicit dependency injection
✅ Configuration-driven
✅ Highly testable (unit + integration)
✅ Loosely coupled services
✅ Production-ready patterns
✅ Easy to mock/stub
✅ Reusable components
✅ Clear separation of concerns
✅ Scalable architecture
✅ Maintainable codebase

---

## Key Architectural Decisions

### 1. Why Service Layer?

**Decision**: Separate business logic into service classes

**Rationale**:
- Node functions should be thin wrappers
- Business logic needs to be testable independently
- Services can be reused across different graphs
- Clear interface contracts

### 2. Why Builder Pattern for Graphs?

**Decision**: Use builder classes instead of module-level graph construction

**Rationale**:
- Enables dependency injection into node closures
- Graph structure becomes inspectable and modifiable
- Can create multiple graph instances with different configs
- Testable graph construction logic

### 3. Why Configuration Dataclasses?

**Decision**: Use dataclasses instead of dict-based config

**Rationale**:
- Type safety and validation
- IDE autocomplete support
- Easy serialization/deserialization
- Integrates with LangGraph's RunnableConfig

### 4. Why Not Class-Based Nodes?

**Decision**: Keep functional nodes, but make them methods of builders

**Rationale**:
- LangGraph is optimized for functional nodes
- Method nodes can access builder's services via `self`
- Maintains simplicity of functional style
- Easier to reason about data flow

### 5. Why Lazy Loading?

**Decision**: Lazy-load expensive resources (LLM, search clients)

**Rationale**:
- Faster application startup
- Only pay for what you use
- Easier to mock in tests
- Resource management flexibility

---

## Anti-Patterns to Avoid

### ❌ God Services

**Problem**: Creating services that do everything

**Solution**: Keep services focused on single responsibility

### ❌ Passing State Through Services

**Problem**: Services modifying state directly

**Solution**: Services return updates, builders handle state

### ❌ Hidden Dependencies

**Problem**: Services depending on other services without injection

**Solution**: Always inject dependencies through constructor

### ❌ Configuration in Code

**Problem**: Hardcoding configuration values

**Solution**: Externalize all configuration

### ❌ Mutable Global State

**Problem**: Shared mutable state across nodes

**Solution**: All state flows through graph state parameter

---

## Next Steps

1. **Review this plan** with team/stakeholders
2. **Create GitHub issues** for each phase
3. **Set up development environment** with new structure
4. **Begin Phase 1** (configuration extraction)
5. **Write tests** as you refactor (test-driven refactoring)
6. **Document** as you go
7. **Iterate** based on learnings

---

## Conclusion

This modularization plan transforms the research assistant from a script-based prototype into a production-ready, maintainable, and scalable application. By introducing clear architectural layers (Configuration → Services → Graphs → Application), eliminating global variables, and embracing dependency injection, the codebase becomes testable, extensible, and ready for real-world deployment.

The phased migration approach allows for incremental refactoring while maintaining functionality at each step. Each phase delivers tangible benefits and can be deployed independently.

**Total Estimated Effort**: 5 weeks (1 developer)
**Risk Level**: Low (incremental refactoring with tests)
**Expected ROI**: High (maintainability, testability, scalability)
