from googletrans import Translator #limited number of translation
import pandas as pd

fileFrom = "Dataset for Detection of Cyber-Trolls.json"
fileTo = "first_iteration.json"

df = pd.read_json(fileFrom, lines=True)

lis=list(df["content"])

print("Translation begin.")

for j in range(len(list(df["content"]))):
    try:
        translator = Translator()
        ls = str(lis[j])
        print(ls)
        tex = translator.translate(ls, src='en', dest='fr').text
        txt = translator.translate(tex, 
                                        src='fr',
                                        dest='en'
                                        ).text
        df["content"][i] = txt
    except:
        print(j)
        break

print("Translation end.")

#print("Rewrite \'content\' begin.")
#for i in len(en):
#    df["content"][i] = en[i]
#print("Rewrite \'content\' end.")

print("Write to file.")
df.to_json(fileTo)