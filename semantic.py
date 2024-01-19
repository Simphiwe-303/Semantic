import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))
print("\n")

# ------------------------------------------------- #

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
	for token2 in tokens:
		print(token1.text, token2.text, token1.similarity(token2))

print("\n")

# -------------------------------------------------- #

sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
	similarity = nlp(sentence).similarity(model_sentence)
	print(sentence + " - ", similarity)

# -------------------------------------------------- #
print("""
# Cat and Banana/Apple have less similarity since a cat don't even eat fruits
# Cat and Monkey have some more similarity because they both are not bipedal
# Monkey and Banana/Apple have a some similarity because a monkey does eat fruits
# Banana and Apple have some similarity since they are fruits.\n""")

print("# NB! In 'en_core_web_md' model if you check similarity from the same list you get some certainty but if you check similarity from two different list you don't get certainty.")
