from wc import count_words

assert count_words("a b c") == 3, count_words("a b c")
assert count_words("a  b\nc") == 3, count_words("a  b\nc")
assert count_words("") == 0, count_words("")
print("all tests passed")
