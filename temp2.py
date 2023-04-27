from ferret import Benchmark
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ferret.explainers.lime import LIMEExplainer
from ferret.explainers.shap import SHAPExplainer

device = 'cuda:0'

name = "Hate-speech-CNERG/hindi-abusive-MuRIL"
model = AutoModelForSequenceClassification.from_pretrained(name).to(device)
tokenizer = AutoTokenizer.from_pretrained(name)

bench = Benchmark(model, tokenizer, explainers=[SHAPExplainer(model, tokenizer),LIMEExplainer(model, tokenizer)])
explanations = bench.explain("Hello, my dog is unfortunate", target=1)
df = bench.get_dataframe(explanations)

print(df)
print(df.columns)
print(len(df))