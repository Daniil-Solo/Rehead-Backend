from llama_cpp import Llama


class DescGenModel:
    def __init__(self, model_path: str):
        self.LLM = Llama(model_path=model_path, n_ctx=2048)

    def preprocess_char(self, text: str):
        pass

    def predict(self, text: str):
        self.prompt = f"""Ты - Лама, русскоговорящая нейросеть, которая помогает генерировать продающие описания для товаров на маркетплейсе основываясь на их характеристиках.
        Напиши на русском языке продающее описание для карточки товара на маркетплейсе. Характеристики товара:
        {text}
        Продающее описание:"""
        output = self.LLM(self.prompt, max_tokens=300)
        return output["choices"][0]["text"]

    def infer(self, text: str):
        pred = self.predict(text)
        return pred
