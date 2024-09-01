import asyncio
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence
import json
from tqdm import tqdm
from ..extras.misc import torch_gc
from ..hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .vllm_engine import VllmEngine


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
        if model_args.infer_backend == "huggingface":
            self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
        elif model_args.infer_backend == "vllm":
            self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
        else:
            raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.engine.start(), self._loop)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, system, tools, image, **input_kwargs), self._loop)
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, system, tools, image, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, image, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        image: Optional["NDArray"] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, image, **input_kwargs):
            yield new_token

    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)

def run_generation(args: Optional[Dict[str, Any]] = None) -> None:
    chat_model = ChatModel(args)
    model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
    print('adapter_name: ',model_args.adapter_name_or_path)
    print("-"*100)
    # dataset_list = ['arguana','climate-fever', 'dbpedia-entity','fever','hotpotqa','nfcorpus','quora','scidocs','scifact','trec-covid','webis-touche2020']
    # dataset_list = ['climate-fever', 'dbpedia-entity','nfcorpus','quora','scidocs','scifact','trec-covid','webis-touche2020']
    dataset_list = ['trec-covid']

    for dataset_name in dataset_list:
        result_list = []
        print(f'now is {dataset_name} dataset')
        json_file = f'data/to_generate_data/{dataset_name}_train_1w.json'
        with open(json_file,'r') as f:
            json_data = json.load(f)
        for data in tqdm(json_data):
            instruction = data['instruction']
            input = data['input']
            cid = data['cid']
            if len(input) <= 3:
                continue
            query = instruction + '\n' + input
            messages= [{"role": "user", "content": query}]
            for _ in range(1):
                try:
                    chat_result = chat_model.chat(messages)[0].response_text
                except Exception as e:
                    print(e)
                    continue
                print(f"dataset:{dataset_name}\ncontent:{chat_result}\n")
                print("-"*100)
                result_list.append({
                        'corpus':input,
                        'pseudo_query':chat_result,
                        'cid':cid
                    })
            if len(result_list) > 8000:
                break
        with open(f'data/{dataset_name}_generate_sft_per1_no_kongge.json', 'w') as f:
            json.dump(result_list, f, indent=4)
        
        
# TODO: 下面的应该是上次主动检索的内
#     json_file = 'data/nq-train_train_1w.json'
#     with open(json_file, 'r') as f:
#         json_data = json.load(f)
#     # prompt_title = 'When you are faced with a question, please follow these guidelines in your response:If you are confident in your answer, start your response with "<No Need>".If you are uncertain and believe additional retrieval is necessary, start your response with "<Need Retrieve>".Do not use any other prefixes or introductory phrases before these tags.Ensure that your responses are clear, structured, and easy to read.'
#     result_list = []
#     import random
#     json_data = random.sample(json_data,1000)
#     for data in tqdm(json_data):
#         for _ in range(1):
#             query = data['output']
#             doc = data['input']
            
#             prompt_query = f'When you are faced with a question, please follow these guidelines in your response:If you are confident in your answer, start your response with "<No Need>".If you are uncertain and believe additional retrieval is necessary, start your response with "<Need Retrieve>".Do not use any other prefixes or introductory phrases before these tags.Ensure that your responses are clear, structured, and easy to read.\
# Examples:\
# Question: What is the capital of France?\
# Answer: <No Need> The capital of France is Paris.\
# Question: What is the population of Mars?\
# Answer: <Need Retrieve> I need to retrieve information to answer that question.\
# Now, follow the same format to answer the following question.\
# Question: {query}\
# Answer:'
#             messages=[{"role": "user", "content": prompt_query}]
#             chat_result = chat_model.chat(messages)[0].response_text
#             print(chat_result)
#             print("-"*100)
#             result_list.append({
#                 'query':query,
#                 'doc':doc,
#                 'pseudo_doc':chat_result
#             })
#     with open('data/nq_our_few_shot.json', 'w') as f:
#         json.dump(result_list, f, indent=4)
        
    

def run_chat() -> None:
    try:
        import platform

        if platform.system() != "Windows":
            import readline  # noqa: F401
    except ImportError:
        print("Install `readline` for a better experience.")

    chat_model = ChatModel()
    messages = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            torch_gc()
            print("History has been removed.")
            continue

        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})
