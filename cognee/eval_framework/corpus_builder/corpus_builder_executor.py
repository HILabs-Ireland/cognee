import cognee
import logging
from typing import Optional, Tuple, List, Dict, Union, Any, Callable, Awaitable

from cognee.eval_framework.corpus_builder.task_getters.TaskGetters import TaskGetters
from cognee.eval_framework.benchmark_adapters.benchmark_adapters import BenchmarkAdapter
from cognee.modules.chunking.TextChunker import TextChunker
from cognee.modules.pipelines.tasks.Task import Task
from cognee.shared.utils import setup_logging


class CorpusBuilderExecutor:
    def __init__(
        self,
        benchmark: Union[str, Any] = "Dummy",
        task_getter: Callable[..., Awaitable[List[Task]]] = None,
    ) -> None:
        if isinstance(benchmark, str):
            try:
                adapter_enum = BenchmarkAdapter(benchmark)
            except ValueError:
                raise ValueError(f"Unsupported benchmark: {benchmark}")
            self.adapter = adapter_enum.adapter_class()
        else:
            self.adapter = benchmark

        self.raw_corpus = None
        self.questions = None
        self.task_getter = task_getter

    def load_corpus(
        self, limit: Optional[int] = None, load_golden_context: bool = False
    ) -> Tuple[List[Dict], List[str]]:
        self.raw_corpus, self.questions = self.adapter.load_corpus(
            limit=limit, load_golden_context=load_golden_context
        )
        return self.raw_corpus, self.questions

    async def build_corpus(
        self,
        limit: Optional[int] = None,
        chunk_size=1024,
        chunker=TextChunker,
        load_golden_context: bool = False,
    ) -> List[str]:
        self.load_corpus(limit=limit, load_golden_context=load_golden_context)
        await self.run_cognee(chunk_size=chunk_size, chunker=chunker)
        return self.questions

    async def run_cognee(self, chunk_size=1024, chunker=TextChunker) -> None:
        setup_logging(logging.ERROR)

        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        await cognee.add(self.raw_corpus)

        tasks = await self.task_getter(chunk_size=chunk_size, chunker=chunker)
        await cognee.cognify(tasks=tasks)
