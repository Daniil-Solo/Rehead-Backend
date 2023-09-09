from llama_cpp import Llama


class DescGenModel:
    def __init__(self, model_path: str):
        self.LLM = Llama(model_path=model_path, n_ctx=2048)

    def predict(self, text: str):
        self.prompt = f"""Привет, Llama-2! Помоги мне создать продающее описание товара для моего магазина на маркетплейсте.
        
        Характеристики товара:
        {text}
        
        Я хочу, чтобы описание было информативным, привлекало внимание покупателей и улучшало SEO для моего товара. Пожалуйста, учти, что текст должен быть на русском языке и не содержать лишних фраз или данных, которые не относятся к товару. Буду благодарен за твою помощь!"""
        output = self.LLM(self.prompt, max_tokens=300)
        return output["choices"][0]["text"]

    def infer(self, text: str):
        pred = self.predict(text)
        return pred
