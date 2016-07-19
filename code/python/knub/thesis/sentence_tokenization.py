from nltk.tokenize import sent_tokenize, word_tokenize
import codecs
import sys
import string

args = sys.argv[1:]
if len(args) != 2:
    print "[binary] articles.txt-file sentences.txt-file"
    sys.exit(1)

articles_file_name = args[0]
sentences_file = codecs.open(args[1], "w", encoding="utf-8")

print "Reading ", articles_file_name
print "Writing ", sentences_file
assert "articles.txt" in articles_file_name
assert "sentences.txt" in sentences_file.name

for line in codecs.open(articles_file_name, 'r', encoding='utf-8'):
    for remove_char in [",", ";", "(", ")", "[", "]", "{", "}", ":", "="]:
        line = line.replace(remove_char, "")
    sentences = sent_tokenize(line)
    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            is_punctuation = all([s in string.punctuation for s in word])
            if not is_punctuation:
                sentences_file.write(word)
                sentences_file.write(" ")
        sentences_file.write("\n")
