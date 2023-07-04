from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, T5ForConditionalGeneration

# Follow instructions on https://python.langchain.com/docs/modules/chains/popular/vector_db_qa
# Revised to drop usage of OpenAI() to use local computing devices
# Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure)

# model_id = 'lmsys/vicuna-7b-v1.3'
# model_id = 'THUDM/chatglm2-6b'
model_id = 'lmsys/fastchat-t5-3b-v1.0'
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print("Tokenizer loaded.")
# model = LlamaForCausalLM.from_pretrained(model_id, device_map='mps').half()
# model = AutoModel.from_pretrained(model_id, device_map='mps', trust_remote_code=True).half()
model = T5ForConditionalGeneration.from_pretrained(model_id, device_map='mps')

# Error printed if using vicuna & AutoModel: The current model class (LlamaModel) is not compatible with `.generate()`, as it doesn't have a language model head
# Error printed if using fastchat & AutoModel: The current model class (T5Model) is not compatible with `.generate()`, as it doesn't have a language model head
print("Model loaded.")

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

local_llm = HuggingFacePipeline(pipeline=pipe)
# 测试模型基础问答能力
# print(local_llm('What is the capital of France? '))

# 加载文件夹中的所有pdf类型的文件
loader = DirectoryLoader('data', glob='**/*.pdf')
# 将数据转成 document 对象，每个文件会作为一个 document
documents = loader.load()

# 初始化加载器
text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=1000, chunk_overlap=0)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)

# 初始化 embeddings 对象， 默认使用sentence_transformers embeddings
embeddings = HuggingFaceEmbeddings()
# 将 document 通过 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
docsearch = Chroma.from_documents(split_docs, embeddings)

# 创建问答对象
qa = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=docsearch.as_retriever())
print('RetrievalQA loaded.')
# 进行问答
result = qa("How to reboot my VM?")
print(result)