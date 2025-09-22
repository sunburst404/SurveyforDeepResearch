### 一些解决办法

https://github.com/langchain-ai/open_deep_research/issues/196
这个人还在尝试用本地的llm使用改deep research

https://github.com/langchain-ai/open_deep_research/issues/166
使用ollma本地部署 deep research，当然，其中也可以找到官方的说明



https://github.com/langchain-ai/open_deep_research/issues/159
✅使用openrouter调用其他api（使用qwen/qwen3-235b-a22b:free可以运行，但是花钱 ） 或者 是其他有效的接口网站和api key
（见**修改方式1**）









### 一些问题

| 时间 | 问题描述                                                     | 解决？                                                       |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 9.20 | 阿里百炼的qwen-plus时，出现格式问题（见**修改方式1**）       | 不支持openai兼容模式，需要改代码。qwen官网说支持dashscope方式，还需要在研究 |
| 9.22 | langchain本地端口打不开，只能用langsmith那个端口，不知道什么意思 |                                                              |
|      |                                                              |                                                              |



### 成功的试验（修改方式）

1. 使用free版本时，有tokens的限制，可以在configuration.py中把token改小一些

   阿里百炼：修改graph.py（加入url）；在env中添加OPENAI_BASE_URL字段，修改api ley；configuration.py中将openai:gpt-4o 改为 openai:YOUR_MODEL
   这里不修改openai字段是经营所得。

   ```python
   writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs, 
           base_url=os.getenv("OPENAI_BASE_URL"))  #graph.py所有该位置
   ```