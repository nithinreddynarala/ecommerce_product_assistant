import asyncio
from utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
import grpc.experimental.aio as grpc_aio
grpc_aio.init_grpc_aio()
model_loader=ModelLoader()

def ensure_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def evaluate_context_precision(query, response, retrieved_context):
    try:
        docs = ensure_list(retrieved_context)
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=docs,
        )

        async def main():
            llm = model_loader.load_llm()
            llm.model_kwargs = {"n": 1} 
            evaluator_llm = LangchainLLMWrapper(llm)
            context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await context_precision.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        return e

def evaluate_response_relevancy(query, response, retrieved_context):
    try:
        docs = ensure_list(retrieved_context)
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=docs,
        )

        async def main():
            llm = model_loader.load_llm()
            llm.model_kwargs = {"n": 1} 
            evaluator_llm = LangchainLLMWrapper(llm)
            embedding_model = model_loader.load_embeddings()
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
            scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
            result = await scorer.single_turn_ascore(sample)
            return result

        return asyncio.run(main())
    except Exception as e:
        return e