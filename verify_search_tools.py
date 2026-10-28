"""Offline tool-contract checks; no Tavily key or network required."""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from search_tools import build_search_tools, months_before, publication_date
from datetime import date


class FakeClient:
    def __init__(self):
        self.calls = []

    async def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {"results": [{
            "url": "https://example.com/programmable-cash",
            "title": "Fixture: programmable cash",
            "content": "Tokenised deposits and programmable money requirements.",
            "published_date": "2026-10-01",
        }]}


async def main():
    client = FakeClient()
    web, news = build_search_tools(client=client, agent_name="needs_gatherer", as_of_date="2026-10-25")
    web_result = await web.ainvoke({"query": "PSP tokenised deposits needs"})
    news_result = await news.ainvoke({"query": "PSP programmable money", "time_range": "month"})
    assert client.calls[0]["topic"] == "general"
    assert "time_range" not in client.calls[0]
    assert client.calls[1]["topic"] == "news"
    assert client.calls[1]["start_date"] == "2026-09-25"
    assert client.calls[1]["end_date"] == "2026-10-26"
    assert client.calls[1]["filter_by_published_date"] is False
    assert "include_domains" not in client.calls[0]
    assert web_result["evidence"][0]["agent"] == "needs_gatherer"
    assert "Tokenised deposits" in web_result["evidence"][0]["excerpt"]
    assert web_result["evidence"][0]["id"] != news_result["evidence"][0]["id"]

    for arguments in ({"query": "   "}, {"query": "cash", "max_results": 21}):
        try:
            await web.ainvoke(arguments)
        except ValidationError:
            pass
        else:
            raise AssertionError("Invalid tool arguments were accepted")
    assert len(client.calls) == 2

    # Verify the tool also runs through LangGraph's worker tool executor.
    tool_graph = StateGraph(MessagesState)
    tool_graph.add_node("tools", ToolNode([web, news]))
    tool_graph.add_edge(START, "tools")
    tool_graph.add_edge("tools", END)
    result = await tool_graph.compile().ainvoke({"messages": [AIMessage(content="", tool_calls=[{
        "name": "search_news", "args": {"query": "PSP cash needs"},
        "id": "fixture-call", "type": "tool_call",
    }])]})
    assert isinstance(result["messages"][-1], ToolMessage)
    assert result["messages"][-1].tool_call_id == "fixture-call"
    assert result["messages"][-1].artifact["evidence"][0]["agent"] == "needs_gatherer"

    class DateClient:
        def __init__(self):
            self.calls = []

        async def search(self, query, **kwargs):
            self.calls.append(kwargs)
            dates = ["2026-07-25", "2026-10-25", "2026-07-24", "2026-10-26",
                     None, "unclear", "Sun, 25 Oct 2026 10:00:00 GMT"]
            return {"results": [{"url": f"https://example.com/{i}", "content": "Fixture",
                                 "published_date": value} for i, value in enumerate(dates)]}

    dated = DateClient()
    _, recent = build_search_tools(client=dated, agent_name="fixture", as_of_date="2026-10-25")
    filtered = await recent.ainvoke({"query": "cash", "time_range": "year"})
    assert dated.calls[0]["start_date"] == "2026-07-25"  # Cannot widen the window.
    assert filtered["excluded_dated_results"] == 2
    assert len(filtered["evidence"]) == 5
    assert sum(e["date_status"] == "uncertain" for e in filtered["evidence"]) == 2
    assert months_before(date(2026, 5, 31), 3) == date(2026, 2, 28)
    assert months_before(date(2024, 5, 31), 3) == date(2024, 2, 29)
    assert publication_date("2026-07") is None
    assert publication_date("2026-10-25T13:00:00Z") == date(2026, 10, 25)

    class FailingClient:
        async def search(self, query, **kwargs):
            raise TimeoutError("Fixture timeout")

    failing_web, _ = build_search_tools(client=FailingClient(), agent_name="fixture")
    try:
        await failing_web.ainvoke({"query": "cash"})
    except TimeoutError:
        pass
    else:
        raise AssertionError("Failure was hidden as a successful search")
    print("PASS: topic routing, three-month news window, uncertain dates, provenance, ToolNode and errors")


if __name__ == "__main__":
    asyncio.run(main())
