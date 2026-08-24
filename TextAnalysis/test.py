negation_words = {'not', 'no', 'nor', 'never', 'none', 'nothing',
                   'neither', 'nowhere', 'cannot'}

def tag_negation(text, negation_words, window=3):
    words = text.split()
    result = []
    neg_counter = 0
    for w in words:
        if w == '!':
            neg_counter = 0
            result.append(w)
            continue
        if w in negation_words:
            neg_counter = window
            result.append(w)
        elif neg_counter > 0:
            result.append(w + '_NEG')
            neg_counter -= 1
        else:
            result.append(w)
    return ' '.join(result)


tests = [
    "not bad",
    "not good",
    "not bad at all really",       # window=3 should stop tagging after 3 words
    "good not bad",                # "good" before "not" should stay untouched
    "never disappointed by this",
    "great taste ! not fresh though",  # "!" should reset the negation scope
]

for t in tests:
    print(f"{t!r:45s} -> {tag_negation(t, negation_words)}")