from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain

# using langchain
template = """For the following question, you will provide a helpful and concise answer in one or two sentences. Provide only the answer with no other formatting. Question: {query} Answer: """

# define the template
prompt = PromptTemplate(
    input_variables=["query"],
    template=template,
)

# load the model
llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    input={"temperature": 2, "max_length": 4000, "top_p": 1},
    verbose=False,
)

# define the chain
chain = LLMChain(llm=llm, prompt=prompt)


def run(input: str) -> str:
    return chain.run({"query": input})
