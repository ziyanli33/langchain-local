# langchain-local
## Introduction
An experimental langchain AI bot using pure local computing resources and documents. 
It by default uses [lmsys/fastchat-t5-3b-v1.0](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) model, which is as tiny as less than 7Gb and fits Macbook Pro M1 well. 

## Usage
### Preparation
1. Install python packages
  ```
  pip3 install -r requirements.txt
  ```
2. Prepare an PDF and put it in `/data`
   > Note: Keep each QA text block within the size of 200 characters to enchance the search results, as the context length of the model is not huge enough.
   
   > Note: Chain the structured titles up together to help the model understand the relationship. For example, use "Title1-Title1.2-Title1.2.4[Text1.2.4]" instead of just "Title1.2.4[Text1.2.4]"
### Run
```
python3 main.py
```



